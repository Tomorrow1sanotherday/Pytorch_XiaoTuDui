import torch
from torch import nn


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
            nn.Linear(64, 10)

  )

    def forward(self,x):
        x = self.model1(x)
        return x


# if __name__ == '__main__':
#     tudui = Tudui()
#
# input = torch.ones(64,3,32,32)
# output = tudui(input)
# print(output.shape)
