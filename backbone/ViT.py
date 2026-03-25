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


class ResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.

        Args:
            block: the basic ResNet block
            num_blocks: the number of blocks per layer
            num_classes: the number of output classes
            nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.return_prerelu = False
        self.device = "cpu"
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.classifier = nn.Linear(nf * 8 * block.expansion, num_classes)

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def set_return_prerelu(self, enable=True):
        self.return_prerelu = enable
        for c in self.modules():
            if isinstance(c, self.block):
                c.return_prerelu = enable


    def GetAllFeaturs(self, x: torch.Tensor):
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        return out_1,out_2,out_3,out_4,feature

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.

        Args:
            block: ResNet basic block
            planes: channels across the network
            num_blocks: number of blocks
            stride: stride

        Returns:
            ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among 'out', 'features', 'both', and 'full')

        Returns:
            output tensor (output_classes)
        """
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        if returnt == 'features':
            return feature

        out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'both':
            return (out, feature)
        elif returnt == 'full':
            return out, [
                out_0 if not self.return_prerelu else out_0_t,
                out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out_4 if not self.return_prerelu else self.layer4[-1].prerelu
            ]

        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both', 'all'] but got {}".format(returnt))


class MyResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.

        Args:
            block: the basic ResNet block
            num_blocks: the number of blocks per layer
            num_classes: the number of output classes
            nf: the number of filters
        """
        super(MyResNet, self).__init__()
        self.return_prerelu = False
        self.device = "gpu"
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        '''
        self.in_planes = nf
        self.layer1_ = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2_ = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3_ = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4_ = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        '''

        self.classifier = nn.Linear(nf * 4 * block.expansion*2, num_classes)

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)
    
    def setNoGrade(self):
        for param in self.parameters():
            param.requires_grad = False

    def set_return_prerelu(self, enable=True):
        self.return_prerelu = enable
        for c in self.modules():
            if isinstance(c, self.block):
                c.return_prerelu = enable

    def ReturnAllFeatures(self, x: torch.Tensor):
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4
        out_4__ = out_4

        out_1 = out_1.view(out_1.size(0), -1)
        out_2 = out_2.view(out_2.size(0), -1)
        out_3 = out_3.view(out_3.size(0), -1)
        out_4 = out_4.view(out_4.size(0), -1)

        feature = avg_pool2d(out_4__, out_4__.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        return out_1,out_2,out_3,out_4,feature

    def GetAllFeaturs(self, x: torch.Tensor):
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4

        feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        return out_1,out_2,out_3,out_4,feature,out_1_,out_2_,out_3_,out_4_,feature_

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.

        Args:
            block: ResNet basic block
            planes: channels across the network
            num_blocks: number of blocks
            stride: stride

        Returns:
            ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among 'out', 'features', 'both', and 'full')

        Returns:
            output tensor (output_classes)
        """
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        '''
        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4
        '''

        #feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        #feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        #totalFeature = torch.cat((feature,feature_),1)

        if returnt == 'features':
            return feature

        out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'both':
            return (out, feature)
        elif returnt == 'full':
            return out, [
                out_0 if not self.return_prerelu else out_0_t,
                out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out_4 if not self.return_prerelu else self.layer4[-1].prerelu
            ]

        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both', 'all'] but got {}".format(returnt))

class PnnNetwork(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """
    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.

        Args:
            block: the basic ResNet block
            num_blocks: the number of blocks per layer
            num_classes: the number of output classes
            nf: the number of filters
        """
        super(PnnNetwork, self).__init__()
        self.return_prerelu = False
        #self.device = "gpu"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        '''
        self.in_planes = nf
        self.layer1_ = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2_ = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3_ = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4_ = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        '''

        self.classifierArr = []
        self.fcArr = []

        self.fc = nn.Linear(nf * 4 * block.expansion*2, 512, device=self.device)
        self.classifier = nn.Linear(512, num_classes)

        self.fcArr.append(self.fc)
        self.classifierArr.append(self.classifier)

        self.adaptiveWeightArr = []
        self.adaptiveWeightArr.append([])
        self.currentExpertIndex = 0

    def GiveTaskSpecificLayer(self,inputs,labels,k):
        x = inputs
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        '''
        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4
        '''

        # feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        # feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        f1 = self.fcArr[k](feature)
        return f1

    def MakePredictionByExpertIndex(self,inputs,labels,k):
        """
                Compute a forward pass.

                Args:
                    x: input tensor (batch_size, *input_shape)
                    returnt: return type (a string among 'out', 'features', 'both', and 'full')

                Returns:
                    output tensor (output_classes)
                """
        x = inputs
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        '''
        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4
        '''

        # feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        # feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        f1 = self.fcArr[k](feature)

        if k == 0:
            out = self.classifierArr[k](f1)
        else:
            softmax = F.softmax(self.adaptiveWeightArr[k])
            sum1 = []
            for i in range(np.shape(softmax)[0]):
                if np.shape(sum1)[0] == 0:
                    sum1 = self.fcArr[i](feature) * softmax[i]
                else:
                    sum1 += self.fcArr[i](feature) * softmax[i]

            myFeature = torch.cat((f1, sum1), 1)
            out = self.classifierArr[k](myFeature)
        return out

    def CreateNewExpert(self):
        nf = self.nf

        for j in range(np.shape(self.fcArr)[0]-1):
            a = self.adaptiveWeightArr[j+1]
            a.requires_grad = False

        for param in self.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(nf * 4 * self.block.expansion * 2, 512, device=self.device)
        self.classifier = nn.Linear(512, self.num_classes)

        self.fcArr.append(self.fc)
        self.classifierArr.append(self.classifier)

        existedN = np.shape(self.fcArr)[0]-1
        self.mynetWeight = nn.Parameter(torch.randn((existedN), requires_grad=True).to(self.device))
        self.adaptiveWeightArr.append(self.mynetWeight)

        self.currentExpertIndex = np.shape(self.fcArr)[0] - 1

        '''
        for param in self.parameters():
            param.requires_grad = False

        self.fc.weight.requires_grad = True
        self.classifier.weight.requires_grad = True
        '''

    def setNoGrade(self):
        for param in self.parameters():
            param.requires_grad = False

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def myprediction(self,x,k):
        with torch.no_grad():
            out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
            if self.return_prerelu:
                out_0_t = out_0.clone()
            out_0 = relu(out_0)
            if hasattr(self, 'maxpool'):
                out_0 = self.maxpool(out_0)

            out_1 = self.layer1(out_0)  # -> 64, 32, 32
            out_2 = self.layer2(out_1)  # -> 128, 16, 16
            out_3 = self.layer3(out_2)  # -> 256, 8, 8
            out_4 = self.layer4(out_3)  # -> 512, 4, 4

            '''
            out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
            out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
            out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
            out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4
            '''

            # feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
            # feature_ = feature_.view(feature_.size(0), -1)  # 512

            feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
            feature = feature.view(feature.size(0), -1)  # 512
            f1 = self.fcArr[k](feature)
            out = self.classifierArr[k](f1)

            return out

    def set_return_prerelu(self, enable=True):
        self.return_prerelu = enable
        for c in self.modules():
            if isinstance(c, self.block):
                c.return_prerelu = enable

    def ReturnAllFeatures(self, x: torch.Tensor):
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4
        out_4__ = out_4

        out_1 = out_1.view(out_1.size(0), -1)
        out_2 = out_2.view(out_2.size(0), -1)
        out_3 = out_3.view(out_3.size(0), -1)
        out_4 = out_4.view(out_4.size(0), -1)

        feature = avg_pool2d(out_4__, out_4__.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        return out_1,out_2,out_3,out_4,feature

    def GetAllFeaturs(self, x: torch.Tensor):
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4

        feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        return out_1,out_2,out_3,out_4,feature,out_1_,out_2_,out_3_,out_4_,feature_

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.

        Args:
            block: ResNet basic block
            planes: channels across the network
            num_blocks: number of blocks
            stride: stride

        Returns:
            ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among 'out', 'features', 'both', and 'full')

        Returns:
            output tensor (output_classes)
        """
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        '''
        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4
        '''

        #feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        #feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        f1 = self.fc(feature)

        #totalFeature = torch.cat((feature,feature_),1)

        if returnt == 'features':
            return feature

        out = self.classifier(f1)

        if returnt == 'out':
            return out
        elif returnt == 'both':
            return (out, feature)
        elif returnt == 'full':
            return out, [
                out_0 if not self.return_prerelu else out_0_t,
                out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out_4 if not self.return_prerelu else self.layer4[-1].prerelu
            ]

        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both', 'all'] but got {}".format(returnt))


    
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
            file="your_model_path" 
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
                param.requires_grad = False   # Freeze vit2 parameters
        
        # Image preprocessing
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

        # Step 1: vit1 full forward to get z1
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
            z1 = x_vit1[:, 0, :]  # class token from full vit1

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
        # Adaptive representation layer and classifier
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

def resnet18(nclasses: int, nf: int = 64) -> ResNet:
    """
    Instantiates a ResNet18 network.

    Args:
        nclasses: number of output classes
        nf: number of filters

    Returns:
        ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)

def mynet(nclasses: int, nf: int = 64):

    """
    Instantiates a ResNet18 network.

    Args:
        nclasses: number of output classes
        nf: number of filters

    Returns:
        ResNet network
    """
    return DynamicRobustDualVit(BasicBlock, [2, 2, 2, 2], nclasses, 768)

    # return MyResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)

    #return PnnNetwork(BasicBlock, [2, 2, 2, 2], nclasses, nf)
