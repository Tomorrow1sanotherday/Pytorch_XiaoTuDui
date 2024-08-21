import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.nn_module import Tudui

dataset = torchvision.datasets.CIFAR10(root = '../dataset/CIFAR10', train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1,0)

    def forward(self, x):
        x = self.conv1(x)
        return x

writer = SummaryWriter("./logs")


step = 0

tudui = Tudui()


for data in dataloader:
    imgs, labels = data
    writer.add_images("in", imgs, step)
    out = tudui(imgs)
    out = torch.reshape(out, (-1, 3, 30, 30))
    writer.add_images("out", out, step)
    step += 1
