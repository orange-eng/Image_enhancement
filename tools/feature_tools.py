# coding: utf-8
 
from keras.models import Model
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation
from pylab import *
import keras
import keras.layers as layer
 
def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col
 
 
def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)
    feature_map_combination = []
    plt.figure()
 
    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)
    print("row:",row)
    print("col:",col)
    layer_num = num_pic
    for i in range(0, num_pic):     #num_pic表示所有的通道数目
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        #plt.subplot(1, row*col, i + 1)
        plt.subplot(row,col,i+1)
        plt.imshow(feature_map_split,cmap='gray')
        axis('off')
        title('feature_map_{}'.format(i))
 
    plt.savefig('./IMG/processing/feature_map/feature_map_table_{}.png'.format(layer_num))
    plt.show()
 
    # # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    print(feature_map_sum.shape)
    plt.imshow(feature_map_sum,cmap='gray')
    plt.savefig("./IMG/processing/feature_map/feature_map_sum_{}.png".format(layer_num))
    plt.show()
 
def create_model():
    model = Sequential()
 
    # 第一层CNN
    # 第一个参数是卷积核的数量，第二三个参数是卷积核的大小
    model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(4, 4)))
 
    # # 第二层CNN
    # model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(3, 3)))
 
    # # 第三层CNN
    # model.add(Convolution2D(9, 5, 5, input_shape=img.shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
 
    # # # 第四层CNN
    # model.add(Convolution2D(4, 3, 3, input_shape=img.shape))
    # model.add(Activation('relu'))
    # # model.add(MaxPooling2D(pool_size=(2, 2)))
 
    return model
 
 
if __name__ == "__main__":
    img = cv2.imread('./IMG/1.png')
    print('origin_size:',img.shape)
    model = create_model()
 
    img_batch = np.expand_dims(img, axis=0)
    print(img_batch.shape)
    conv_img = model.predict(img_batch)  # conv_img 卷积结果
 
    visualize_feature_map(conv_img)
