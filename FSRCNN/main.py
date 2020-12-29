import numpy as np
import tensorflow as tf
import pprint
from model import SRCNN

import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))

#----------------------------------------一种初始化所有参数的方法
flags = tf.app.flags
#第一个是参数名称，第二个参数是默认值，第三个是参数描述
flags.DEFINE_integer("epoch",1,"Number of epoch[3]")
flags.DEFINE_integer("batch_size",4,"the size of batch [128]")
flags.DEFINE_integer("image_size",33,"the size of image [33]")
flags.DEFINE_integer("label_size",21,"The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

#--------------------------------------------------------
PP = pprint.PrettyPrinter()

def main(_):
    PP.pprint(flags.FLAGS.__flags)

    if not os.path.exists(path+".\\"+FLAGS.checkpoint_dir):
        os.makedirs(path+".\\"+FLAGS.checkpoint_dir)
    if not os.path.exists(path+".\\"+FLAGS.sample_dir):          #如果目录中没有samle文件夹，则创建一个
        os.makedirs(path+".\\"+FLAGS.sample_dir)

    with tf.Session() as sess:
        srcnn = SRCNN(sess = sess,
        image_size=FLAGS.image_size,
        label_size=FLAGS.label_size,
        batch_size=FLAGS.batch_size,
        c_dim=FLAGS.c_dim,
        checkpoint_dir=FLAGS.checkpoint_dir
        )
        srcnn.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()

