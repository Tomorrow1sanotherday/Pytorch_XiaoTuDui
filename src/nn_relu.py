import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.read_data import MyDataset

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = input.reshape([-1, 1, 2, 2])


dataset = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

class My_relu(torch.nn.Module):
    def __init__(self):
        super(My_relu, self).__init__()
        self.relu1 = torch.nn.ReLU()
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid1(x)
        return x


my_relu = My_relu()

writter = SummaryWriter(log_dir='./logs')

step = 0
for data in dataloader:
    imgs, labels = data
    writter.add_images("input", imgs, step)
    outputs = my_relu(imgs)
    writter.add_images("output", outputs, step)
    step += 1

writter.close()

