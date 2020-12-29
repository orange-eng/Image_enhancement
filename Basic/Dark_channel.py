import cv2 as cv
import numpy as np
import matplotlib.pyplot as pyplot
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))


def darkchannel():
  for i in range(0,rows-1):
    for j in range(0,cols-1):
      min_rgb = img_arr[i][j][0]
      if min_rgb  > img_arr[i][j][1]:
        min_rgb = img_arr[i][j][1]
      elif min_rgb  > img_arr[i][j][2]:
        min_rgb = img_arr[i][j][2]
      for c in range(channels):
        data[i][j][c] = min_rgb

def min_filter():
  for i in range(0, rows):
    for j in range(0, cols):
      for c in range(0, channels):
        if i == 0 or j == 0 or i == rows - 1 or j == cols - 1:
          min_data[i][j][c] = data[i][j][c]
        elif j == 0:
          min_data[i][j][c] = data[i][j][c]
        else:
          min = 255
          for m in range(i - 2, i + 2):
            for n in range(j - 2, j + 2):
              if min > data[m][n][c]:
                min = data[m][n][c]
          min_data[i][j][c] = min


def guided_filter():
  #I:导向图  p:输入图
  # q = a * I + b
  I = img_arr
  mean_I = cv.blur(I, (r, r))  # I的均值平滑
  mean_p = cv.blur(data, (r, r))  # p的均值平滑
  mean_II = cv.blur(I*I, (r, r))  # I*I的均值平滑
  mean_Ip = cv.blur(I*data, (r, r))  # I*p的均值平滑
  var_I = mean_II - mean_I * mean_I  # 方差
  cov_Ip = mean_Ip - mean_I * mean_p  # 协方差
  a = cov_Ip / (var_I +eps)
  b = mean_p - a *mean_I
  mean_a = cv.blur(a, (r, r))  # 对a、b进行均值平滑
  mean_b = cv.blur(b, (r, r))
  q = mean_a*img_arr + mean_b
  return q


def select_bright(data,V):
  order = [0 for i in range(size)]
  m = 0
  for t in range(0,rows):
    for j in range(0,cols):
      order[m] = data[t][j][0]
      m = m+1
  order.sort(reverse=True)
  index =int(size * 0.001) #从暗通道中选取亮度最大的前0.1%
  mid = order[index]
  A = 0
  img_hsv = cv.cvtColor(img_copy,cv.COLOR_RGB2HLS)
  for i in range(0,rows):
    for j in range(0,cols):
      if data[i][j][0]>mid and img_hsv[i][j][1]>A:
        A = img_hsv[i][j][1]
  V = V * w
  print(V)
  t = 1 - V/A
  t = np.maximum(t,t0)
  return t,A


# def img_repair():
#   J = np.zeros(img_arr.shape)
#   for i in range(0,rows):
#     for j in range(0,cols):
#       for c in range(0,channels):
#         J[i][j][c] = (img_arr[i][j][c]-A/255.0)/t[i][j][c]+A/255.0
#   return J


r = 75  # kernel
eps = 0.001
w = 0.95
t0 = 0.10
img = cv.imread(path+"\\1.bmp")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("original",img)
img_copy = img
img = img/255.0 # 归一化
img_arr = np.array(img)
rows, cols, channels = img.shape
data = np.empty([rows,cols,channels],dtype=float)
min_data = np.empty([rows, cols, channels],dtype=float)
darkchannel()
min_filter()
V = guided_filter()
size = rows * cols
t, A = select_bright(data, V)
pyplot.imshow(min_data)
pyplot.axis("off")
pyplot.show()

def repair():
  J = np.zeros(img_arr.shape)
  for i in range(0,rows):
    for j in range(0,cols):
      for c in range(0,channels):
        t[i][j][c] = t[i][j][c]-0.25 # 不知道为什么这里减掉0.25效果才比较好
        J[i][j][c] = (img_arr[i][j][c]-A/255.0)/t[i][j][c]+A/255.0
  return J
J = repair()
pyplot.imshow(J)
pyplot.axis("off")
pyplot.show()

cv.waitKey(0)
cv.destroyAllWindows()