# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F
import torch

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer
import torchattacks
import torch.nn as nn
import copy
import numpy as np

class Dynamicmodel(ContinualModel):
    NAME = 'dynamicmodel'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Dark Experience Replay.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')

        parser.add_argument('--r_alpha', type=float, required=True,
                            help='Penalty weight.')

        return parser

    def __init__(self, backbone, loss, args, transform):
        super(Dynamicmodel, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device)

        self.adversary = torchattacks.FGSM(self.net, eps=8 / 255)
        self.mse_loss = nn.MSELoss()

    def SetCurrentExpertByIndex(self,k):
        self.net.currentExpertIndex = k
        self.net.fc = self.net.fcArr[k]
        self.net.classifier = self.net.classifierArr[k]

    def myPrediction(self,x,k):
        with torch.no_grad():
            #Perform the prediction according to the seloeced expert
            out = self.net.myprediction(x,k)
        return out

    def end_task(self, dataset):
        self.net.CreateNewExpert()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        self.opt = self.get_optimizer()
        self.adversary = torchattacks.FGSM(self.net, eps=8 / 255)
        self.buffer = Buffer(self.args.buffer_size)

        #self.oldNet = copy.deepcopy(self.net)
        #self.oldNet.setNoGrade()

    def Calculate_Regularization(self,inputs,labels):
        sum1 = 0
        for i in range(np.shape(self.net.classifierArr)[0]-1):
            output1 = self.net.MakePredictionByExpertIndex(inputs,labels,i)
            output2 = self.oldNet.MakePredictionByExpertIndex(inputs,labels,i)
            sum1 += F.mse_loss(output1, output2)
        sum1 = sum1 / int(np.shape(self.net.classifierArr)[0]-1)
        #sum1 *= 0.1
        sum1 *= self.args.r_alpha
        return sum1

    def Calculate_FeatureRegularization(self,inputs,labels):
        sum1 = 0
        for i in range(np.shape(self.net.classifierArr)[0] - 1):
            output1 = self.net.GiveTaskSpecificLayer(inputs, labels, i)
            output2 = self.oldNet.GiveTaskSpecificLayer(inputs, labels, i)
            sum1 += F.mse_loss(output1, output2)
        sum1 = sum1 / int(np.shape(self.net.classifierArr)[0] - 1)
        # sum1 *= 0.1
        sum1 *= self.args.r_alpha
        return sum1


    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        tot_loss = 0

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss += loss.item()

        '''
        #Add noise to the new data
        new_inputs, new_labels = inputs.cuda(), labels.cuda()  # add this line
        aSamples = self.adversary(new_inputs, new_labels)
        aOutputs = self.net(aSamples)
        loss2 = self.args.alpha * self.loss(aOutputs, new_labels)
        loss2.backward()
        tot_loss += loss2.item()
        #End the noise
        '''

        if not self.buffer.is_empty():
            buf_inputs, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_outputs = self.net(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

            '''
            buf_labels = torch.argmax(buf_logits,1)
            buf_inputs, buf_labels = buf_inputs.cuda(), buf_labels.cuda()  # add this line
            aSamples = self.adversary(buf_inputs, buf_labels)
            aOutputs = self.net(aSamples)
            loss_mse2 = self.args.alpha * F.mse_loss(aOutputs, buf_logits)
            loss_mse2.backward()
            tot_loss += loss_mse2.item()
            '''

            '''
            if self.current_task != 0:
                rloss = self.Calculate_Regularization(buf_inputs,buf_labels) + self.Calculate_FeatureRegularization(buf_inputs,buf_labels)
                rloss.backward()
                tot_loss += rloss.item()

                rloss2 = self.Calculate_Regularization(aSamples, buf_labels)+ self.Calculate_FeatureRegularization(buf_inputs,buf_labels)
                rloss2.backward()
                tot_loss += rloss2.item()
                
            '''

        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        return tot_loss
