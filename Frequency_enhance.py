import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))



def enhance(f):
    rows, cols = f.shape
    f_mask = np.zeros(f.shape)
    original = np.zeros(f.shape)

    # 中心化预处理
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            original[i, j] = f[i, j] * ((-1) ** (i + j))

    F = np.fft.fft2(original)
    # 生成高通滤波模板
    D0 = 20
    for i in range(rows):
        for j in range(cols):
            temp = (i - rows / 2) ** 2 + (j - cols / 2) ** 2
            f_mask[i, j] = 1 - np.e ** (-1 * temp / (2 * D0 ** 2))

    return np.abs(np.fft.ifft2(F * f_mask))

def show(f, s, a, b, c):
    plt.subplot(a, b, c)
    plt.imshow(f, "gray")
    plt.axis('on')
    plt.title(s)
    

original=cv.imread(path+'./img/1.jpg', 0)
f=enhance(original)
plt.figure()
show(original, "original", 2, 2, 1)
show(f, "shape", 2, 2, 2)
ret, dst = cv.threshold(f + original, 255, 255, cv.THRESH_TRUNC)
show(dst, "enhance", 2, 2, 3)
plt.show()