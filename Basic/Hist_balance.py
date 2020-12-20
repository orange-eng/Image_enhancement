import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

# 绘制直方图函数
def grayHist(img):
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBins = 256
    histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                      facecolor='black', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()
 
img = cv.imread(path+'./img/1.jpg', 0)
out = 2.0 * img
# 进行数据截断，大于255的值截断为255
out[out > 255] = 255
# 数据类型转换
out = np.around(out)
out = out.astype(np.uint8)
# 分别绘制处理前后的直方图
grayHist(img)
grayHist(out)
cv.imshow("img", img)
cv.imshow("out", out)
cv.waitKey(0)
cv.destroyAllWindows()
