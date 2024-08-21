import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)




class Tudui(torch.nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,x):
        x = self.maxpool1(x)
        return x


tudui = Tudui()

writer = SummaryWriter(log_dir='logs')
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()

