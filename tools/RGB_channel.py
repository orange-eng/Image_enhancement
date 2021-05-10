import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


img = cv2.imread('IMG/butterfly.png')

#cv2.imshow("original",img)


def get_red(img):
    redImg = img[:,:,2]
    return redImg
def get_green(img):
    greenImg = img[:,:,1]
    return greenImg
def get_blue(img):
    blueImg = img[:,:,0]
    return blueImg
def save_png(img,img_name):
    cv2.imwrite(img_name,img)
b = get_blue(img)
g = get_green(img)
r = get_red(img)

srcImage_new = cv2.merge([r, g, b])

cv2.imshow("Blue 2", b)
cv2.imshow("Green 2", g)
cv2.imshow("Red 2", r)

b_plt = get_blue(srcImage_new)
plt.imshow(b,cmap='gray')
#plt.imshow(srcImage_new)

plt.show()


cv2.waitKey (0)  
cv2.destroyAllWindows() 

'''
save_png(b,'IMG/processing/blue.png')
save_png(g,'IMG/processing/green.png')
save_png(r,'IMG/processing/red.png')
print(b)
data = pd.DataFrame(b)
writer = pd.ExcelWriter('B.xlsx')
data.to_excel(writer,'page_1',float_format='%.5f')
writer.save()
writer.close()

data = pd.DataFrame(g)
writer = pd.ExcelWriter('B.xlsx')
data.to_excel(writer,'page_2',float_format='%.5f')
writer.save()
writer.close()

data = pd.DataFrame(r)
writer = pd.ExcelWriter('B.xlsx')
data.to_excel(writer,'page_3',float_format='%.5f')
writer.save()
writer.close()


'''