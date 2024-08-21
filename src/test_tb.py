from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer = SummaryWriter('logs')
img_path = "../dataset/practice/val/bees/6a00d8341c630a53ef00e553d0beb18834-800wi.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
writer.add_image('test2', img_array, global_step=2, dataformats='HWC')





writer.close()
