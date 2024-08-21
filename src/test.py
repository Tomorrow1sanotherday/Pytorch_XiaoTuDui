import PIL
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *

image_path = '../imgs/飞机.jpg'
img = PIL.Image.open(image_path)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                           torchvision.transforms.ToTensor()])

img = transform(img)
print(img.shape)

model = torch.load('tudui_9.0.pth')

writer = SummaryWriter(log_dir='log')
img = torch.reshape(img, (1,3,32,32))
writer.add_images('img',img,global_step=0)
writer.close()
img = img.cuda()
model.eval()
with torch.no_grad():
    output = model(img)
print(output.argmax(dim=1, keepdim=True))

