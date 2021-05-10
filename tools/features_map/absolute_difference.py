import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))
import torch.nn.functional as F
import torch

def grey_scale(img_gray ):
    rows, cols = img_gray.shape
    flat_gray = img_gray.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    print('A = %d,B = %d' % (A, B))
    output = np.uint8(255 / (B - A) * (img_gray - A) + 0.5)
    return output
 

picture_name = "head"
img_1=cv2.imread(path + '\\{}.png'.format(picture_name))
img_2=cv2.imread(path + '\\{}_bicubic.png'.format(picture_name))
img=img_1-img_2
#result=grey_scale(img[0])      #图片归一化
cv2.imwrite(path + '\\{}_absolute_1.png'.format(picture_name),img)
