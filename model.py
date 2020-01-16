import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class ClassifierModel(nn.Module):
  def __init__(self):
    super(ClassifierModel, self).__init__()
    self.conv = torch.nn.Conv2d(3, 6, 5)
    self.conv2 = torch.nn.Conv2d(6, 16, 5)

    self.maxpool = torch.nn.MaxPool2d(2, 2)

    self.layer1 = torch.nn.Linear(400, 120)
    self.layer2 = torch.nn.Linear(120, 84)
    self.layer3 = torch.nn.Linear(84, 10)

  def forward(self, x):
    x = F.relu(self.conv(x))
    x = self.maxpool(x)
    x = F.relu(self.conv2(x))
    x = self.maxpool(x)

    x = x.view(-1, 400)
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = self.layer3(x)
    return x
