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
首先获得所有bmp文件的文件路径，存储再data里面


## model.py


## main.py
