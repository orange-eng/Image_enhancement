import os
import math
import pickle
import random
import numpy as np
import glob
import torch
import cv2
import sys
sys.path.append("..")
import util as util
path = os.path.abspath(os.path.dirname(sys.argv[0]))

#we first downsample the original images with scaling factors 0.6, 0.7, 0.8, 0.9 to generate the HR/LR images.
for scale in [1, 0.9, 0.8, 0.7, 0.6]:
    GT_folder = path + '/data/DIV2K800_GT'
    save_GT_folder = path + '/data/DIV2K800_scale/GT'
    for i in [save_GT_folder]:
        if os.path.exists(i):
            pass
        else:
            os.makedirs(i)
    img_GT_list = util._get_paths_from_images(GT_folder)
    for path_GT in img_GT_list:
        img_GT = cv2.imread(path_GT)
        img_GT = img_GT
        # imresize

        rlt_GT = util.imresize_np(img_GT, scale, antialiasing=True)
        print(str(scale) + "_" + os.path.basename(path_GT))

        cv2.imwrite(os.path.join(save_GT_folder, str(scale) + "_" + os.path.basename(path_GT)), rlt_GT)
