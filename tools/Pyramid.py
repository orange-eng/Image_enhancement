import cv2
import numpy as np

def gaussian_pyr(img,lev):
    img = img.astype(np.float)
    g_pyr = [img]
    cur_g = img
    for index in range(lev):
        cur_g = cv2.pyrDown(cur_g)
        g_pyr.append(cur_g)
    return g_pyr




def laplacian_pyr(img,lev):
    img = img.astype(np.float)
    g_pyr = gaussian_pyr(img,lev)
    l_pyr = []
    for index in range(lev):
        cur_g = g_pyr[index]
        next_g = cv2.pyrUp(g_pyr[index+1])
        cur_l = cv2.subtract(cur_g,next_g)
        l_pyr.append(cur_l)
    l_pyr.append(g_pyr[-1])
    return l_pyr

if __name__ == "__main__":
    img = cv2.imread('./result/cat/cat.jpg')
    print('origin_size:',img.shape)
    pyramid = laplacian_pyr(img,4)
    print(pyramid[3].shape)
    cv2.imshow('Pyramid',pyramid[3])
    
    cv2.waitKey(0)

def lpyr_recons(l_pyr):
    lev = len(l_pyr)
    cur_l = l_pyr[-1]
    for index in range(lev-2,-1,-1):
        print(index)
        cur_l = cv2.pyrUp(cur_l)
        next_l = l_pyr[index]
        cur_l = cur_l + next_l
    return cur_l