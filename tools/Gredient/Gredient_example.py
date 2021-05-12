# -*- coding:utf-8 -*-
# Author : MMagicLoren
# @Email : 993983320@qq.com
# @Time : 2019/10/13 15:59
# @File : 图像梯度计算.py
# @Project : Workspace
import cv2 as cv
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

 
def sobel_demo(image):
    grad_x = cv.Sobel(src, cv.CV_64F, 1, 0, ksize=3)
    gradx = cv.convertScaleAbs(grad_x)
    grad_y = cv.Sobel(src, cv.CV_64F, 0, 1, ksize=3)
    grady = cv.convertScaleAbs(grad_y)
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv.imshow("grad_x", grad_x)  # 将src图片放入该创建的窗口中
    cv.imshow("grad_y", grad_y)  # 将src图片放入该创建的窗口中
    cv.imshow("gradxy", gradxy)  # 将src图片放入该创建的窗口中
 
 
if __name__ == '__main__':
    src = cv.imread(path + '\\1.jpg')  # 读入图片放进src中
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)  # 创建窗口, 窗口尺寸自动调整
    cv.imshow("input image", src)
    sobel_demo(src)
 
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()
