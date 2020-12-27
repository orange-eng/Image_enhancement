# SRCNN

## 前言
* 这是CNN应用在超分辨率领域的开山之作，在学习的过程中遇到很多问题，在这里也有必要好好总结一下，希望可以对大家有所帮助

**SRCNN**

SRCNN首先使用双三次(bicubic)插值将低分辨率图像放大成目标尺寸，接着通过三层卷积网络拟合非线性映射，最后输出高分辨率图像结果。本文中，作者将三层卷积的结构解释成三个步骤：图像块的提取和特征表示，特征非线性映射和最终的重建。
三个卷积层使用的卷积核的大小分为为9x9,，1x1和5x5，前两个的输出特征个数分别为64和32。用Timofte数据集（包含91幅图像）和ImageNet大数据集进行训练。使用均方误差(Mean Squared Error, MSE)作为损失函数，有利于获得较高的PSNR。

------
## 目录（后期会根据学习内容增加）

[所需环境 Environment](#所需环境)

[tools](#tools.py)

[models](#models.py)

[main](#main.py)

## 所需环境
* Anaconda3（建议使用）
* python3.6.6
* VScode 1.50.1 (IDE)
* pytorch 1.3 (pip package)
* torchvision 0.4.0 (pip package)
* numpy 1.19.4
* tensorflow 1.13.2


## tools.py

- imsave(image,path)  存储图像image到指定目录path
```python
def imsave(image,path):
    return scipy.misc.imsave(path,image)
```

- imread(path,is_graysacle) 通过scipy方式读取文件（scipy版本不能太高，1.2.1）
```python
#返回值是ndarry格式的图片
def imread(path,is_grayscale = True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        return scipy.misc.imread(path,flatten=True,mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path,mode='YCbCr').astype(np.float)
```
- modecrop(image, scale=3)输入一个图像，根据scale的大小，把宽和高调整成为scale的整数倍
```python
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
```

- read_data  读取h5文件的数据，其中包括两个dataset: data和label
```python
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
```

- preprocess(path,scale=3)  对图片进行模糊操作。
先正则化，把所有像素变为0~1之间
将图像先缩小为原来的1/scale倍，然后再扩大scale倍，得到的input_是模糊图像，label_是清晰图像
```python
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
    #scipy.ndimage.interpolation.zoom这种放大图像的方法叫做最临近插值算法
    #prefilter : bool, optional 。参数预滤波器确定输入是否在插值之前使用spline_filter进行预过滤（对于> 1的样条插值所必需的）。 如果为False，则假定输入已被过滤。 默认为True。
    return input_, label_
```

- prepare_data(sess,dataset): 准备数据集
如果是训练状态，则选择训练集
如果为测试状态，则选择测试集
dataset是一个文件路径，里面装有train和test两个文件夹
返回值为所有bmp文件的文件路径data={序号:文件路径}
```python
def prepare_data(sess, dataset):
    """
    Args:
        dataset: choose train dataset or test dataset

        For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    if FLAGS.is_train:
        filenames = os.listdir(dataset)
        data_dir = os.path.join(path, dataset)
        data = glob.glob(os.path.join(data_dir, "*.bmp"))   #读取所有后缀为.bmp的文件，data存着所有bmp的文件路径
    else:
        data_dir = os.path.join(os.sep, (os.path.join(path, dataset)), "Set5")
        data = glob.glob(os.path.join(data_dir, "*.bmp"))   #读取所有后缀为.bmp的文件，data存着所有bmp的文件路径

    return data
```

- make_data(sess,data,label) 制作h5文件，用于存储训练图像数据或是测试图像数据
首先找到H5文件所在的路径，然后写入两个dataset：data和label
```python
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
```

- merge(images,size) 拼接操作
size表示[nx,ny]，即横向和纵向的格子数目
先是第一行开始排列格子，再试下一行，以此类推
i 就是对每一行的格子进行便编号
j 就是对每一列的格子进行编号
```python
def merge(images,size):
    h,w = images.shape[1],images.shape[2]
    img = np.zeros((h*size[0],w*size[1],1))   #size[1]为横向的格子数目 size[0]为纵向格子数目
    for idx,image in enumerate(images):
        i = idx % size[1]     #取余数 ，就是对横向的格子进行编号
        # j = idx % size[1]   
        j = idx // size[1]    #整除 得到列数
        # 先是第一行开始排列格子，再试下一行，以此类推
        img[j*h:j*h+h,i*w:i*w+w,:] = image
    return img
```

- input_setup(sess,config) 最重要的函数
首先读取图像，然后生成各自的子图像（[33,33]或[21,21]）,保存为H5格式的文件
```python
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
```

## model.py
- def __init__初始化所有参数并传入
```python
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
```

- def build_model(self) 初始化模型的所有参数，损失函数采用MSE
```python
def build_model(self):

    self.images = tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.c_dim],name='images')
    self.labels = tf.placeholder(tf.float32,[None,self.label_size,self.label_size,self.c_dim],name='labels')
    #第一层CNN：对输入图片的特征提取。（9 x 9 x 64卷积核）
    #第二层CNN：对第一层提取的特征的非线性映射（1 x 1 x 32卷积核）
    #第三层CNN：对映射后的特征进行重建，生成高分辨率图像（5 x 5 x 1卷积核）
    #权重
    self.weights = {
        'w1': tf.Variable(tf.random_normal([9,9,1,64],stddev=1e-3),name='w1'),
        'w2': tf.Variable(tf.random_normal([1,1,64,32],stddev=1e-3),name='w2'),
        'w3': tf.Variable(tf.random_normal([5,5,32,1],stddev=1e-3),name='w3'),
    }
    self.biases = {
        'b1': tf.Variable(tf.zeros([64]),name = 'b1'),
        'b2': tf.Variable(tf.zeros([32]),name = 'b2'),
        'b3': tf.Variable(tf.zeros([1]),name = 'b3')
    }
    self.pred = self.model()
    #Loss function(MSE)
    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
    self.saver = tf.train.Saver()   #an overview of variables, saving and restoring.
```
- def model(self) 整个卷积神经网络在这里构建
训练步骤可以分为以下几个：
** 1.定义一个优化器        self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
** 2.初始化所有参数        tf.initialize_all_variables().run()
** 3.进入迭代过程          for ep in xrange(config.epoch):
** 4.以batch为单位，提取指定数量的图片   batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
** 5.开始训练               _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
```python
def model(self):
    conv1 = tf.nn.relu(tf.nn.conv2d(self.images,self.weights['w1'],strides=[1,1,1,1],padding='VALID')+self.biases['b1'])        #输入是四位向量 kernel*kernel*input*output
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1,self.weights['w2'],strides=[1,1,1,1],padding='VALID')+self.biases['b2'])
    conv3 = tf.nn.conv2d(conv2,self.weights['w3'],strides=[1,1,1,1],padding='VALID')+self.biases['b3']
    return conv3
```

- def train(self, config) 模型训练函数（比较复杂）
```python
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
        for ep in xrange(config.epoch):
            # Run by batch images
            batch_idxs = len(train_data) // config.batch_size
            for idx in xrange(0, batch_idxs):
                batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
                batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]
                counter += 1
                _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
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
```


## main.py

- PP = pprint.PrettyPrinter() 可以展示数据更加工整，相当于print的升级版
可以对比如下代码，即可发现差异
```python
import pprint
data=['generate_csv\\train_00.csv','generate_csv\\train_01.csv',
      'generate_csv\\train_02.csv', 'generate_csv\\train_03.csv',
      'generate_csv\\train_10.csv', 'generate_csv\\train_11.csv']
print(data)
print("--------分界线--------------")
pprint.pprint(data)
```

- flags = tf.app.flags 初始化所有的变量，非常实用的方式，值得学习
```python  
flags = tf.app.flags
#第一个是参数名称，第二个参数是默认值，第三个是参数描述
flags.DEFINE_integer("epoch",1,"Number of epoch[3]")
flags.DEFINE_integer("batch_size",128,"the size of batch [128]")
flags.DEFINE_integer("image_size",33,"the size of image [33]")
flags.DEFINE_integer("label_size",21,"The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
FLAGS = flags.FLAGS
```
如果要在其他地方使用的话，需要采用如下方式
```python
FLAGS = tf.app.flags.FLAGS
```

- main(_) 这里把main 也写成一个函数，可以学习这种写代码的方式，感觉更加简洁
- os函数可以用于创建文件夹，也很实用
```python
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
```
- Session提供了Operation执行和Tensor求值的环境。如下面所示，
```python
import tensorflow as tf
# Build a graph.
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a * b
# Launch the graph in a session.
sess = tf.Session()
# Evaluate the tensor 'c'.
print sess.run(c)
sess.close()
# result: [3., 8.]
```
