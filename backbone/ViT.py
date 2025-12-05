# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu
from timm import create_model
import torchvision.transforms as transforms
from backbone.VAEmodels.vanilla_vae import *

from backbone import MammothBackbone
#from backbone.vit import *
import copy
from scipy.stats import entropy

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.

    Args:
        in_planes: number of input channels
        out_planes: number of output channels
        stride: stride of the convolution

    Returns:
        convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.

        Args:
            in_planes: the number of input channels
            planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.return_prerelu = False
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, input_size)

        Returns:
            output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        if self.return_prerelu:
            self.prerelu = out.clone()

        out = relu(out)
        return out

class DynamicRobustDualVit(MammothBackbone):
    def __init__(self, block: BasicBlock, num_blocks: List[int], num_classes: int, nf: int) -> None:
        super(DynamicRobustDualVit, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.nf = nf    # Feautre dimension (ViT class token size)
        model_name = 'vit_base_patch16_224'
        
        self.vit1 = create_model(
            model_name,
            pretrained = True,
            num_classes = self.num_classes,  
            pretrained_cfg_overlay=dict(
            file="your vit model path" 
            )
        ).to(self.device)
        
        for param in self.vit1.parameters():
            param.requires_grad = False
        for i in range(-3, 0):
            for param in self.vit1.blocks[i].parameters():
                param.requires_grad = True
                
        self.vit2_blocks = nn.ModuleList([
            copy.deepcopy(self.vit1.blocks[i]).to(self.device) for i in range(-3, 0)
        ])
        for block in self.vit2_blocks:
            for param in block.parameters():
                param.requires_grad = False   
        
        self.vit_process = transforms.Compose([transforms.Resize((224, 224))])
        
        self.prob_classifier = nn.Sequential(
            nn.Linear(2 * self.nf, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.prob_list = nn.ModuleList()
        self.prob_list.append(self.prob_classifier)
        
        self.expert_modules = nn.ModuleList()
        self.current_task_index = 0
        self.add_expert()  # Initialize first expert


    def MakePredictionByExpertIndex(self, x, labels, k, returnt='out'):
        x = self.vit_process(x)
        x_patch = self.vit1.patch_embed(x)  # (B, N, 768)
        cls_token = self.vit1.cls_token.expand(x_patch.size(0), -1, -1)  # (B, 1, 768)
        x_patch = torch.cat((cls_token, x_patch), dim=1)  # (B, N+1, 768)
        x_patch = x_patch + self.vit1.pos_embed
        x_patch = self.vit1.pos_drop(x_patch)

        x_vit1 = x_patch
        for blk in self.vit1.blocks:
            x_vit1 = blk(x_vit1)
        z1 = x_vit1[:, 0, :]  # (B, 768)

        with torch.no_grad():
            x_vit2 = x_patch
            for blk in self.vit1.blocks[:-3]:
                x_vit2 = blk(x_vit2)
            for blk in self.vit2_blocks:
                x_vit2 = blk(x_vit2)
            z2 = x_vit2[:, 0, :]  # (B, 768)

        combined_z = torch.cat([z1, z2], dim=-1)  # (B, 1536)
        probs = self.prob_list[k](combined_z)  # (B, 2)
        p1, p2 = probs[:, 0:1], probs[:, 1:2]

        z = z1 * p1 + z2 * p2  
        
        if len(self.expert_modules) > 1 and labels is not None:
            mi_weights = self.compute_mi_weights(z, labels)
            if mi_weights is not None:
                z_historical = []
                for i, expert in enumerate(self.expert_modules[:-1]):
                    z_h = expert[0](z)
                    z_historical.append(z_h * mi_weights[i])
                z = z + sum(z_historical)

        out = self.expert_modules[k](z)
        if returnt == 'features':
            return z
        if returnt == 'out':
            return out

    def myprediction(self, x, labels, k):
        with torch.no_grad():
            x = self.vit_process(x)
            x_patch = self.vit1.patch_embed(x)  # (B, N, 768)
            cls_token = self.vit1.cls_token.expand(x_patch.size(0), -1, -1)  # (B, 1, 768)
            x_patch = torch.cat((cls_token, x_patch), dim=1)  # (B, N+1, 768)
            x_patch = x_patch + self.vit1.pos_embed
            x_patch = self.vit1.pos_drop(x_patch)

            x_vit1 = x_patch
            for blk in self.vit1.blocks:
                x_vit1 = blk(x_vit1)
            z1 = x_vit1[:, 0, :]  

            x_vit2 = x_patch
            for blk in self.vit1.blocks[:-3]:
                x_vit2 = blk(x_vit2)
            for blk in self.vit2_blocks:
                x_vit2 = blk(x_vit2)
            z2 = x_vit2[:, 0, :]  # class token from vit2_blocks

            combined_z = torch.cat([z1, z2], dim=-1)  # (B, 2 * nf)
            probs = self.prob_list[k](combined_z)  # (B, 2)
            p1, p2 = probs[:, 0:1], probs[:, 1:2]  # (B, 1)

            z = z1 * p1 + z2 * p2  
            
            if len(self.expert_modules) > 1 and labels is not None:
                mi_weights = self.compute_mi_weights(z, labels)
                if mi_weights is not None:
                    z_historical = []
                    for i, expert in enumerate(self.expert_modules[:-1]):
                        z_h = expert[0](z)  
                        z_historical.append(z_h * mi_weights[i])
                    z = z + sum(z_historical)

            out = self.expert_modules[k](z)
            return out
        
    def add_expert(self):
        adaptive_layer = nn.Sequential(
            nn.Linear(self.nf, 512),
            nn.ReLU(),
            nn.Linear(512, 768),
        )
        classifier = nn.Linear(768, self.num_classes, device=self.device)
        expert = nn.Sequential(adaptive_layer, nn.ReLU(), classifier).to(self.device)
        self.expert_modules.append(expert)

        self.prob_classifier = nn.Sequential(
            nn.Linear(2 * self.nf, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.prob_list.append(self.prob_classifier)


    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, task_idx: int = None,
                return_features: bool = False, return_vit_features: bool = False):
        if task_idx is None:
            task_idx = self.current_task_index

        x = self.vit_process(x)  

        x_patch = self.vit1.patch_embed(x)  # (B, N, 768)
        cls_token = self.vit1.cls_token.expand(x_patch.size(0), -1, -1)  # (B, 1, 768)
        x_patch = torch.cat((cls_token, x_patch), dim=1)  # (B, N+1, 768)
        x_patch = x_patch + self.vit1.pos_embed
        x_patch = self.vit1.pos_drop(x_patch)

        x_vit1 = x_patch
        for blk in self.vit1.blocks:
            x_vit1 = blk(x_vit1)
        z1 = x_vit1[:, 0, :]  

        with torch.no_grad():
            x_vit2 = x_patch
            for blk in self.vit1.blocks[:-3]:
                x_vit2 = blk(x_vit2)
            for blk in self.vit2_blocks:
                x_vit2 = blk(x_vit2)
            z2 = x_vit2[:, 0, :] 

        combined_z = torch.cat([z1, z2], dim=-1)  # (B, 1536)
        probs = self.prob_list[task_idx](combined_z)  
        p1, p2 = probs[:, 0:1], probs[:, 1:2]  # (B, 1)

        z = z1 * p1 + z2 * p2  

        if len(self.expert_modules) > 1 and labels is not None:
            mi_weights = self.compute_mi_weights(z, labels)
            if mi_weights is not None:
                z_historical = []
                for i, expert in enumerate(self.expert_modules[:-1]):
                    z_h = expert[0](z)  
                    z_historical.append(z_h * mi_weights[i])
                z = z + sum(z_historical)  

        out = self.expert_modules[task_idx](z)

        if return_vit_features:
            return z1, z2
        if return_features:
            return z
        return out

    def set_task(self, task_idx: int):
        self.current_task_index = task_idx

    def to(self, device, **kwargs):
        self.device = device
        self.vit1.to(device)
        self.vit2_blocks.to(device)
        self.prob_list.to(device)
        self.expert_modules.to(device)

        return super().to(device, **kwargs)
    
    def estimate_mutual_information(self, features, labels):
        """Estimate mutual information between features and labels."""
        batch_size = features.size(0)
        features = features.view(batch_size, -1).cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        features_discrete = np.digitize(features, bins=np.linspace(features.min(), features.max(), 20))
        joint_dist = np.histogram2d(features_discrete[:, 0], labels, bins=20)[0] / batch_size
        feature_dist = np.sum(joint_dist, axis=1)
        label_dist = np.sum(joint_dist, axis=0)
        mi = entropy(feature_dist, base=2) + entropy(label_dist, base=2) - entropy(joint_dist.flatten(), base=2)
        return max(mi, 0.0)  # Ensure non-negative MI

    def compute_mi_weights(self, z1, labels):
        """Compute MI-based weights for historical experts."""
        if  len(self.expert_modules) <= 1:
            return None
        mi_scores = []
        for expert in self.expert_modules[0:-1]:
            features = expert[0](z1)  # Adaptive layer output
            mi = self.estimate_mutual_information(features, labels)
            mi_scores.append(mi)
        mi_scores = torch.tensor(mi_scores, device=self.device)
        weights = torch.softmax(mi_scores / 0.3, dim=0)
        return 0.1 * weights


def mynet(nclasses: int, nf: int = 64):

    return DynamicRobustDualVit(BasicBlock, [2, 2, 2, 2], nclasses, 768)
