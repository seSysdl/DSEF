import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
import copy
import numpy as np
from utils.buffer import Buffer
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from models.utils.continual_model import ContinualModel
from scipy.stats import entropy

class SiameseViT(ContinualModel):
    NAME = 'siamesevit'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning with adversarial robustness.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, default=1.0, help='Adversarial loss weight.')
        parser.add_argument('--r_alpha', type=float, default=0.1, help='Regularization weight for MMD.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(SiameseViT, self).__init__(backbone, loss, args, transform)
        self.loss = loss
        self.args = args
        self.transform = transform
        self.buffer = Buffer(self.args.buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.adversary = torchattacks.FGSM(self.net, eps=8 / 255)
        self.mse_loss = nn.MSELoss()
        self.opt = self.get_optimizer()
        
        self.old_net = None

    def get_optimizer(self):
        return torch.optim.Adam(self.net.parameters(), lr=0.001)

    def SetCurrentExpertByIndex(self, k):
        self.net.current_task_index = k
    
    def myPrediction(self, x, labels, k):
        with torch.no_grad():
            out = self.net.myprediction(x, labels, k)
        return out
    
    def compute_mmd(self, x, y, gamma=None):
        """Compute MMD loss between x and y using an efficient RBF kernel implementation."""
        def pairwise_distances_sq(x, y):
            """Efficient squared Euclidean distance computation."""
            x_norm = (x**2).sum(1, keepdim=True)
            y_norm = (y**2).sum(1, keepdim=True)
            return x_norm - 2 * torch.mm(x, y.t()) + y_norm.t()

        if gamma is None:
            gamma = 1.0 / (2 * x.shape[1])  # Heuristic for gamma

        # Compute pairwise squared distances
        xx = pairwise_distances_sq(x, x)
        yy = pairwise_distances_sq(y, y)
        xy = pairwise_distances_sq(x, y)

        # Compute RBF kernel values
        k_xx = torch.exp(-gamma * xx).mean()
        k_yy = torch.exp(-gamma * yy).mean()
        k_xy = torch.exp(-gamma * xy).mean()

        return k_xx + k_yy - 2 * k_xy

    def Calculate_Regularization(self, inputs, labels):
        sum1 = 0
        for i in range(len(self.net.expert_modules) - 1):
            output1 = self.net.MakePredictionByExpertIndex(inputs, labels, i)
            output2 = self.old_net.MakePredictionByExpertIndex(inputs, labels, i)
            sum1 += self.compute_mmd(output1, output2)
        sum1 = sum1 / int(len(self.net.expert_modules) - 1)
        return sum1
    
    def Calculate_FeatureRegularization(self, inputs, labels):
        sum2 = 0
        for i in range(len(self.net.expert_modules) - 1):
            feature1 = self.net.MakePredictionByExpertIndex(inputs, labels, i, returnt='features')
            feature2 = self.old_net.MakePredictionByExpertIndex(inputs, labels, i, returnt='features')
            sum2 += self.compute_mmd(feature1, feature2)
        sum2 = sum2 / int(len(self.net.expert_modules) - 1)
        return sum2

    def end_task(self, dataset):
        """Save current expert and prepare for new task."""
        self.net.add_expert()
        self.net.set_task(self.net.current_task_index + 1)
        self.buffer = Buffer(self.args.buffer_size)  # Reset buffer
        self.opt = self.get_optimizer()
        
        # update ViT2
        self.net.vit2_blocks = nn.ModuleList([
            copy.deepcopy(self.net.vit1.blocks[i]).to(self.device) for i in range(-3, 0)
        ])
        for block in self.net.vit2_blocks:
            for param in block.parameters():
                param.requires_grad = False   # Freeze vit2 parameters
        
        self.old_net = copy.deepcopy(self.net)
        for param in self.old_net.vit1.parameters():
            param.requires_grad = False

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """Training step for a batch."""
        self.opt.zero_grad()
        tot_loss = 0.0

        # Standard forward pass
        outputs = self.net(inputs, labels)
        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss += loss.item()

        # Adversarial samples
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        inputs = inputs.clone().detach().requires_grad_(True)
        adv_inputs = self.adversary(inputs, labels)
        adv_outputs = self.net(adv_inputs, labels)
        adv_loss = self.args.alpha * self.loss(adv_outputs, labels)
        adv_loss.backward()
        tot_loss += adv_loss.item()

        # Buffer rehearsal
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            
            buf_outputs = self.net(buf_inputs, buf_labels)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

            # buf_labels = torch.argmax(buf_logits, dim=1)
            buf_inputs, buf_labels = buf_inputs.to(self.device), buf_labels.to(self.device)
            buf_adv_inputs = self.adversary(buf_inputs, buf_labels)
            buf_adv_outputs = self.net(buf_adv_inputs, buf_labels)
            loss_mse_adv = self.args.alpha * F.mse_loss(buf_adv_outputs, buf_logits)
            loss_mse_adv.backward()
            tot_loss += loss_mse_adv.item()

            if self.current_task != 0:
                reg_loss = self.args.r_alpha * (self.Calculate_Regularization(buf_inputs, buf_labels) + self.Calculate_FeatureRegularization(buf_inputs, buf_labels))
                reg_loss.backward()
                tot_loss += reg_loss.item()

                reg_loss2 = self.args.r_alpha * (self.Calculate_Regularization(buf_adv_inputs, buf_labels) + self.Calculate_FeatureRegularization(buf_adv_inputs, buf_labels))
                reg_loss2.backward()
                tot_loss += reg_loss2.item()

        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)
        return tot_loss
    