import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

dataset = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=False, download=True,
                                       transform=transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)


class Tudui(torch.nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.liner1 = torch.nn.Linear(196608, 10)

    def forward(self, x):
        x = self.liner1(x)
        return x

tudui = Tudui()

for data in dataloader:
    imgs, label = data
    print(imgs.size())
    output = torch.flatten(imgs)
    output = tudui(output)
    print(output.size())