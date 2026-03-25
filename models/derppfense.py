# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
import torchattacks
import torch

#####################
class Derppfense(ContinualModel):
    NAME = 'derppfense'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Dark Experience Replay++.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        self.adversary = torchattacks.FGSM(self.net, eps=8 / 255)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()

        outputs = self.net(inputs)

        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss = loss.item()

        # Add loss to the adversarial examples
        new_inputs, new_labels = inputs.cuda(), labels.cuda()  # add this line
        aSamples = self.adversary(new_inputs, new_labels)
        aOutputs = self.net(aSamples)
        loss2 = self.loss(aOutputs, new_labels)
        loss2.backward()
        tot_loss += loss2.item()
        # End loss to the adversarial example

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            buf_outputs = self.net(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            buf_outputs = self.net(buf_inputs)
            loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
            loss_ce.backward()
            tot_loss += loss_ce.item()

            #one_hot_labels = one_hot_labels.to(device)
            #buf_inputs = buf_inputs.to(device)
            buf_inputs, buf_labels = buf_inputs.cuda(), buf_labels.cuda()  # add this line
            aSamples = self.adversary(buf_inputs,buf_labels)
            aOutputs = self.net(aSamples)

            loss_mse2 = self.args.alpha * F.mse_loss(aOutputs, buf_logits)
            loss_mse2.backward()
            tot_loss += loss_mse2.item()

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return tot_loss
