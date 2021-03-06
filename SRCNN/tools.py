
import h5py
import numpy as np
from imageio import imread
import scipy.ndimage
import glob

import tensorflow as tf
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

try:
  xrange
except:
  xrange = range

FLAGS = tf.app.flags.FLAGS
def read_data(path):
    """
    Read h5 format data file
    
    Args:
        path: file path of desired file
        data: '.h5' file format that contains train data values
        label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label

def preprocess(path, scale=3):
    """
    Preprocess single image file 
        (1) Read original image as YCbCr format (and grayscale as default)
        (2) Normalize
        (3) Apply image file with bicubic interpolation

    Args:
        path: file path of desired file
        input_: image applied bicubic interpolation (low-resolution)
        label_: image with original resolution (high-resolution)
    """
    image = imread(path, is_grayscale=True)
    label_ = modcrop(image, scale)

    # Must be normalized
    image = image / 255.
    label_ = label_ / 255.

    input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
    input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)

    return input_, label_
#--------------------------------------------------------把图片的长和宽都变成scale=3的整数倍
def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
    
    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image
#----------------------------------------------读取图片，变为float格式
def imread(path,is_grayscale = True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        return scipy.misc.imread(path,flatten=True,mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path,mode='YCbCr').astype(np.float)
#-------------------------------------------------准备数据

def prepare_data(sess, dataset):
    """
    Args:
        dataset: choose train dataset or test dataset
        
        For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    if FLAGS.is_train:
        filenames = os.listdir(dataset)
        data_dir = os.path.join(path, dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(path, dataset)), "Set5")
        data = glob.glob(os.path.join(data_dir, "*.bmp"))

    return data

def make_data(sess, data, label):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if FLAGS.is_train:
        savepath = os.path.join(path, 'checkpoint/train.h5')
    else:
        savepath = os.path.join(path, 'checkpoint/test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)

#--------------------------------------------------
def input_setup(sess, config):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    读取图像文件并生成它们的子图像，并将它们保存为h5文件格式。
    """
    # Load data path
    if config.is_train:
        data = prepare_data(sess, dataset=path+"\\Train")
    else:
        data = prepare_data(sess, dataset=path+"\\Test")

    sub_input_sequence = []     #全部为[33*33]
    sub_label_sequence = []     #全部为[21*21]
    padding = abs(config.image_size - config.label_size) / 2 # 6

    if config.is_train:
        for i in xrange(len(data)):                             #一幅图作为一个data
            input_, label_ = preprocess(data[i], config.scale)  #得到data[]的LR和HR图input_和label_

            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape

            #把input_和label_分割成若干自图sub_input和sub_label
            for x in range(0, h-config.image_size+1, config.stride):
                for y in range(0, w-config.image_size+1, config.stride):
                    sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
                    sub_label = label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size] # [21 x 21]
                    #按image size大小重排 因此 imgae_size应为33 而label_size应为21
                    sub_input = sub_input.reshape([config.image_size, config.image_size, 1])        #增加一个通道，变成[33*33*1]    
                    sub_label = sub_label.reshape([config.label_size, config.label_size, 1])        #增加一个通道，变成[21*21*1]

                    sub_input_sequence.append(sub_input)                #在sub_input_sequence末尾加sub_input中元素 但考虑为空
                    sub_label_sequence.append(sub_label)

    else:
        input_, label_ = preprocess(data[2], config.scale)

        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape

        # Numbers of sub-images in height and width of image are needed to compute merge operation.
        nx = ny = 0 
        for x in range(0, h-config.image_size+1, config.stride):
            nx += 1; ny = 0
            for y in range(0, w-config.image_size+1, config.stride):
                ny += 1
                sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
                sub_label = label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size] # [21 x 21]
                
                sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
                sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
    arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]

    make_data(sess, arrdata, arrlabel)
    #制作train.h5文件或是test.h5文件，用于存储训练图像数据或是测试图像数据
    #首先找到H5文件所在的路径，然后写入两个dataset：data和label

    if not config.is_train:
        return nx, ny

#--------------------------------------------------存储图像

def imsave(image,path):
    return scipy.misc.imsave(path,image)
# #--------------------------------------------------
# 拼接操作
def merge(images,size):
    h,w = images.shape[1],images.shape[2]
    img = np.zeros((h*size[0],w*size[1],1))
    for idx,image in enumerate(images):
        i = idx % size[1]
        # j = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h,i*w:i*w+w,:] = image
    return img








