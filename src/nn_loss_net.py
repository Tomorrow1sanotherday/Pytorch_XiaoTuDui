from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

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

 )

    def forward(self, x):
        x = self.model1(x)
        return x


tudui = Tudui()

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(tudui.parameters(), lr=0.001, momentum=0.9)

for data in dataloader:
    imgs, labels = data
    outputs = tudui(imgs)
    result_loss = loss(outputs, labels)
    print(result_loss.item())
    result_loss.backward()
    print("ok")
    optimizer.step()
    optimizer.zero_grad()
    print("ok")

writer = SummaryWriter("logs")
writer.add_graph(tudui, input)
writer.close()



