# ESPCN
ESPCN论文 [paper](https://arxiv.org/pdf/1609.05158.pdf)


## Prerequisites
 * Python 3.6
 * TensorFlow
 * Numpy
 * Scipy version > 0.18

## 使用方法
1. 把训练集直接放入images文件中（jpg格式即可）
2. 运行prepare_data.py文件，即可生成指定文件夹，并把训练集分割成很多个小块，用于训练；并生成每一个图片的文件名
3. 运行train.py文件，即可训练网络
4. 运行python generate.py文件，即可生成超分辨率图像
模型的参数都存放在json文件中，调参可以修改里面的参数即可
