import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

import cv2 as cv
import numpy as np

#------------------------------------------------Model
inputs = keras.Input(shape=(128,128,3),name='img')

x=layers.Conv2D(
    filters=64,             #卷积神经元数目
    kernel_size=9,          #感受野大小
    padding='same',         #不改变图像大小
    activation=tf.nn.relu   #激活函数
)(inputs)

x=layers.Conv2D(
    filters=32,             #卷积神经元数目
    kernel_size=1,          #感受野大小
    padding='same',         #不改变图像大小
    activation=tf.nn.relu   #激活函数
)(x)

outputs = layers.Conv2D(
    filters=3,             #卷积神经元数目
    kernel_size=5,          #感受野大小
    padding='same',         #不改变图像大小
)(x)

model = keras.Model(inputs=inputs,outputs=outputs,name='SRCNN_Model')

model.summary()


#----------------------------------------数据集
ishape = 128
 
# load data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
 
# 缩小数据集,控制内存占用(以下win10，8G内存可用)
train_image = train_images[0:1000]      #设置训练数目
test_image  = test_images[0:10]
 
X_train = np.array([cv.resize(i,(ishape,ishape), interpolation=cv.INTER_NEAREST) for i in train_image]) / 255
X_test  = np.array([cv.resize(i,(ishape,ishape), interpolation=cv.INTER_NEAREST) for i in test_image]) / 255
 
y_train = np.array([cv.resize(i,(ishape,ishape), interpolation=cv.INTER_CUBIC) for i in train_image]) / 255
y_test  = np.array([cv.resize(i,(ishape,ishape), interpolation=cv.INTER_CUBIC) for i in test_image]) / 255

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                loss='mse',
                metrics=['mae']     #编译
)

test_scores = model.evaluate(X_test,y_test,verbose=2)   #激活

print('Test loss:', test_scores[0])
print('Test mae:', test_scores[1])
 
# Save entire model to a HDF5 file
model.save(path+'.\\param\\SRCNN.h5')

#-----------------------------------开始应用

ishape = 128
print(1)
img = cv.imread(path+'\\car.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img,(ishape,ishape), interpolation=cv.INTER_NEAREST)# (36,36,3)->(128,128,3)
cv.imshow("img",img)

img = np.reshape(img,(1,ishape,ishape,3)) / 255.
 
# 处理图像超分辨率
img_SR = model.predict(img)
 
cv.imshow('img_SR',img_SR[0])
cv.waitKey(0)
cv.destroyAllWindows()
