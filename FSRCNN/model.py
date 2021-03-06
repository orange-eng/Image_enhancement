from tools import(
    read_data,
    input_setup,
    imsave,
    merge
)

import time
import os
import matplotlib.pyplot as plt 
import numpy
import tensorflow as tf 

import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

try:
  xrange
except:
  xrange = range

class SRCNN(object):

    def __init__(self, 
                sess, 
                image_size=33,
                label_size=21, 
                batch_size=128,
                c_dim=1, 
                checkpoint_dir=None, 
                sample_dir=None):
        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()
    #---------------------------------------------------初始化所有系数，w和b
    def build_model(self):
        
        self.images = tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.c_dim],name='images')
        self.labels = tf.placeholder(tf.float32,[None,self.label_size,self.label_size,self.c_dim],name='labels')
        self.weights = {
            'w1': tf.Variable(tf.random_normal([5,5,1,56],stddev=1e-3),name='w1'),
            'w2': tf.Variable(tf.random_normal([1,1,56,12],stddev=1e-3),name='w2'),
            'w3_4': tf.Variable(tf.random_normal([3,3,12,12],stddev=1e-3),name='w3_4'),            
            'w4': tf.Variable(tf.random_normal([1,1,12,56],stddev=1e-3),name='w4'),
            'w5': tf.Variable(tf.random_normal([9,9,56,1],stddev=1e-3),name='w5'),
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([56]),name = 'b1'),
            'b2': tf.Variable(tf.zeros([12]),name = 'b2'),
            'b3_4': tf.Variable(tf.zeros([12]),name = 'b3_4'),
            'b4': tf.Variable(tf.zeros([56]),name = 'b4'),
            'b5': tf.Variable(tf.zeros([1]),name = 'b5')
        }
        self.pred = self.model()
        #Loss function(MSE)
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.saver = tf.train.Saver()   #an overview of variables, saving and restoring.

#-----------------------------------------------------训练过程
    def train(self, config):
        if config.is_train:
            input_setup(self.sess, config)
        else:
            nx, ny = input_setup(self.sess, config)
        if config.is_train:     
            data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
        else:
            data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")
        train_data, train_label = read_data(data_dir)
        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
        tf.initialize_all_variables().run()
        counter = 0
        start_time = time.time()

        if config.is_train:
            print("Training...")
            for ep in xrange(config.epoch):     #进入迭代过程
                # Run by batch images
                batch_idxs = len(train_data) // config.batch_size   #求出所有batch的个数，以batch为单位
                for idx in xrange(0, batch_idxs):
                    batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
                    batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]
                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
                    #feed_dict={y:3}表示把3赋值给y
                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                        % ((ep+1), counter, time.time()-start_time, err))

        else:
            print("Testing...")
            result = self.pred.eval({self.images: train_data, self.labels: train_label})

            result = merge(result, [nx, ny])
            result = result.squeeze()
            #image_path = os.path.join(path, config.sample_dir)
            image_path = os.path.join(path, "result\\test_image2.png")
            #image_path = path + "\\result\\test1.png"
            imsave(result, image_path)        

#-----------------------------------------------------模型构建
    def model(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images,self.weights['w1'],strides=[1,1,1,1],padding='VALID')+self.biases['b1'])        #输入是四位向量 kernel*kernel*input*output
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1,self.weights['w2'],strides=[1,1,1,1],padding='VALID')+self.biases['b2'])
        
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2,self.weights['w3_4'],strides=[1,1,1,1],padding='SAME')+self.biases['b3_4'])
        conv4 = tf.nn.relu(tf.nn.conv2d(conv3,self.weights['w3_4'],strides=[1,1,1,1],padding='SAME')+self.biases['b3_4'])
        conv5 = tf.nn.relu(tf.nn.conv2d(conv4,self.weights['w3_4'],strides=[1,1,1,1],padding='SAME')+self.biases['b3_4'])
        conv6 = tf.nn.relu(tf.nn.conv2d(conv5,self.weights['w3_4'],strides=[1,1,1,1],padding='SAME')+self.biases['b3_4'])

        conv7 = tf.nn.relu(tf.nn.conv2d(conv6,self.weights['w4'],strides=[1,1,1,1],padding='SAME')+self.biases['b4'])
        conv8 = tf.nn.conv2d(conv7,self.weights['w5'],strides=[1,1,1,1],padding='VALID')+self.biases['b5']
        return conv8

        




