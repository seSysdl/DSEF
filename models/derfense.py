# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer
import torchattacks
import numpy as np
import torch
import torch.nn as nn

class DerFense(ContinualModel):
    NAME = 'derfense'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Dark Experience Replay.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(DerFense, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        self.adversary = torchattacks.FGSM(self.net, eps=8 / 255)

        self.mse_loss = nn.MSELoss()

    def CalculateDistance(self,a1,a2):
        return self.mse_loss(a1,a2)

    def Calculate_DistanceBetwenTwoSamples(self,a1,a2):
        out1, out2, out3, out4, features = self.net.ReturnAllFeatures(a1)
        out1_, out2_, out3_, out4_, features_ = self.net.ReturnAllFeatures(a2)

        d1 = self.CalculateDistance(out1, out1_)
        d2 = self.CalculateDistance(out2, out2_)
        d3 = self.CalculateDistance(out3, out3_)
        d4 = self.CalculateDistance(out4, out4_)
        d5 = self.CalculateDistance(features, features_)
        sumd = d1 + d2 + d3 + d4 + d5
        return sumd

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()
        tot_loss = 0

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss += loss.item()

        #Add loss to the adversarial examples
        new_inputs, new_labels = inputs.cuda(), labels.cuda()  # add this line
        aSamples = self.adversary(new_inputs, new_labels)
        aOutputs = self.net(aSamples)
        loss2 = self.loss(aOutputs, new_labels)
        loss2.backward()
        tot_loss += loss2.item()
        #End loss to the adversarial example

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()

            num_classes = np.shape(buf_logits)[1]
            labels = torch.argmax(buf_logits,1)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            #one_hot_labels = one_hot_labels.to(device)
            #buf_inputs = buf_inputs.to(device)
            buf_inputs, labels = buf_inputs.cuda(), labels.cuda()  # add this line
            aSamples = self.adversary(buf_inputs,labels)
            aOutputs = self.net(aSamples)

            loss_mse2 = self.args.alpha * F.mse_loss(aOutputs, buf_logits)
            loss_mse2.backward()

            #Regulairzation loss
            #sumd = self.Calculate_DistanceBetwenTwoSamples(buf_inputs,aSamples)
            #sumd.backward()

            tot_loss += loss_mse.item() + loss_mse2.item()

        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return tot_loss
