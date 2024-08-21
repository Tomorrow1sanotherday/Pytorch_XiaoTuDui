import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms


test_data = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# img, target = test_data[0]
# print(img.shape)
# print(target)
# print(test_loader)

writer = SummaryWriter('./logs')
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images('test_data_drop',imgs, step)
    step += 1

writer.close()