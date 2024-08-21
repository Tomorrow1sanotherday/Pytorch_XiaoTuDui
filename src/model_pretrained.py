import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch import nn


# train_data = torchvision.datasets.ImageNet(root='../dataset/ImageNet', split = 'train', download=True, transform=transforms.ToTensor())
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

train_data = torchvision.datasets.CIFAR10(root='../dataset/CIFAR10', train=True, download=True, transform=torchvision.transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

vgg16_true.classifier.add_module('add_liner', nn.Linear(in_features=1000, out_features=10))

print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(in_features=4096, out_features=10)
print(vgg16_false)

