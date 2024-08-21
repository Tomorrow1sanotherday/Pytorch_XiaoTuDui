import cv2
from torch.utils.tensorboard import SummaryWriter
from torchvision import  transforms
img_path = "../dataset/practice/val/bees/6a00d8341c630a53ef00e553d0beb18834-800wi.jpg"
writer = SummaryWriter(log_dir='logs')
img = cv2.imread(img_path)

trans_totensor = transforms.ToTensor()
trans_resize = transforms.Resize((512, 512))
trans_normalize = transforms.Normalize([1, 3, 5],[0.5,0.5,0.5])
trans_corp = transforms.RandomCrop(422)
trans_compose = transforms.Compose([trans_totensor,trans_resize, trans_normalize])
img = trans_compose(img)
for i in range(10):
    img_corp = trans_corp(img)
    writer.add_image('img_corp', img_corp, i)

writer.close()
