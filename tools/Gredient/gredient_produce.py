
import cv2 as cv
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

def sobel_demo(image):
    grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)  # 采用Scharr边缘更突出,Sobel改为Scharr即可
    grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)

    gradx = cv.convertScaleAbs(grad_x)  # 由于算完的图像有正有负，所以对其取绝对值
    grady = cv.convertScaleAbs(grad_y)

    # 计算两个图像的权值和，dst = src1*alpha + src2*beta + gamma
    gradxy = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    
    gradx = cv.applyColorMap(gradx, cv.COLORMAP_JET)     #此处的三通道热力图是cv2使用GBR排列
    grady = cv.applyColorMap(grady, cv.COLORMAP_JET)     #此处的三通道热力图是cv2使用GBR排列
    gradxy = cv.applyColorMap(gradxy, cv.COLORMAP_JET)     #此处的三通道热力图是cv2使用GBR排列

    cv.imshow("gradx", gradx)
    cv.imshow("grady", grady)
    cv.imshow("gradient", gradxy)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("gradx.jpg", gradx)
    cv.imwrite("grady.jpg", grady)
    cv.imwrite("gradxy.jpg", gradxy)
print(cv.__version__)
img=cv.imread(path + '\\2.jpg',cv.COLOR_BGR2GRAY)
sobel_demo(img)


