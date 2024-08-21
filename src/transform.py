from torchvision import transforms
import cv2
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
img_path = "../dataset/practice/val/ants/8124241_36b290d372.jpg"
img = Image.open(img_path)
cv_img = cv2.imread(img_path)
print(cv_img)
writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(cv_img)

writer.add_image('test', tensor_img, 0)
writer.close()

