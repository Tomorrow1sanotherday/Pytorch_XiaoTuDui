from torch import nn
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=False, download=True)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
 )

    def forward(self, x):
        x = self.model1(x)
        return x


tudui = Tudui()
print(tudui)
input = torch.ones((64,3,32,32))
output = tudui(input)

writer = SummaryWriter("logs")
writer.add_graph(tudui, input)
writer.close()



