import torch.nn as nn
import torch
import torchvision

import torch.nn.functional as F
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class PretrainedResnet50(nn.Module):
    def __init__(self):
        super(PretrainedResnet50, self).__init__()
        self.pretrained = torchvision.models.resnet50(pretrained='imagenet')
        self.out_size = self.pretrained.fc.in_features
        self.pretrained.fc = Identity()
        conv_1_weights = self.pretrained.conv1.weight.data
        self.pretrained.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pretrained.conv1.weight.data = torch.zeros(size=self.pretrained.conv1.weight.data.shape)
        #print(self.pretrained.conv1.weight.data.shape)
        self.pretrained.conv1.weight.data[:, :3] = conv_1_weights[:,:3]

        self.pen_norm = nn.BatchNorm1d(self.out_size)
        self.pen_layer = nn.Linear(self.out_size, 2048)
        self.final_norm = nn.BatchNorm1d(2048)
        self.final_activ = nn.PReLU(2048)
        self.final_layer = nn.Linear(2048, 28)

    def forward(self, inputs):
        x = self.pretrained(inputs)
        x = self.pen_norm(x)
        x = F.dropout(x, 0.5)
        x = self.pen_layer(x)
        x = self.final_activ(x)
        x = self.final_norm(x)
        x = F.dropout(x, 0.5)
        x = self.final_layer(x)
        return x
