import torchvision

from PIL import Image
from torch.utils.tensorboard import SummaryWriter

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root = '../dataset/CIFAR10', train=True, transform=transform, download=True)
test_set = torchvision.datasets.CIFAR10(root = '../dataset/CIFAR10', train=False, transform=transform, download=True)

# print(test_set[0])
# print(test_set.classes)
# img, target = test_set[8]
# img.show()
# print(target)
writter = SummaryWriter(log_dir='./logs')
for i in range(10):
    image, label = train_set[i]
    writter.add_image('test2',image, label)
writter.close()