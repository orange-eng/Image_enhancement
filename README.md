# Image_enhancement

## 前言
* 这里是orange研究inage enhancement进行整理总结，总结的同时也希望能够帮助更多的小伙伴。后期如果有学习到新的知识也会与大家一起分享。

------
## 目录（后期会根据学习内容增加）
* basic
    * 直方图均衡化（已完成）
    * Retinex_SSR（已完成）
    * Retinex_MSR（已完成）
    * Frequency_enhance（已完成）
* SRCNN（已完成）
* SRGAN（已完成）
* ESPCN（已完成）
* LapSRN(已完成)
### 目录

[所需环境 Environment](#所需环境)

[SRCNN](#SRCNN)

[SRGAN](#SRGAN)

[ESPCN](#ESPCN)

[LapSRN](#LapSRN)

## 所需环境
* Anaconda3（建议使用）
* python3.6.6
* VScode 1.50.1 (IDE)
* pytorch 1.3 (pip package)
* torchvision 0.4.0 (pip package)
* numpy 1.19.4
* tensorflow 1.13.2

## SRCNN
1. SRCNN
SRCNN是深度学习用在超分辨率重建上的开山之作。SRCNN的网络结构非常简单，仅仅用了三个卷积层，网络结构如下图所示。

SRCNN首先使用双三次(bicubic)插值将低分辨率图像放大成目标尺寸，接着通过三层卷积网络拟合非线性映射，最后输出高分辨率图像结果。本文中，作者将三层卷积的结构解释成三个步骤：图像块的提取和特征表示，特征非线性映射和最终的重建。
三个卷积层使用的卷积核的大小分为为9x9,，1x1和5x5，前两个的输出特征个数分别为64和32。用Timofte数据集（包含91幅图像）和ImageNet大数据集进行训练。使用均方误差(Mean Squared Error, MSE)作为损失函数，有利于获得较高的PSNR。
code: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html


## SRGAN
- CVPR2017，把生成对抗网络与SR相结合

1. 数据集
- 高清图片数据集DIV
链接：https://pan.baidu.com/s/15DZSXEK7_XVnsSbpojeESA
提取码：aslw
复制这段内容后打开百度网盘手机App，操作更方便哦--来自百度网盘超级会员V5的分享

2. 使用
- 把高清图片数据集放在dataset/train文件下面，即可开始训练
- 注意修改好文件路径
- Disiscriminator training有些问题，在loss函数部分。这部分先注释掉了，这意味着Discriminator不可被训练（这个问题还没有解决）

## ESPCN
- CVPR2016， 创新点在于亚卷积层的使用。即进行最后一次卷积操作之后，对每一个像素点进行重新排列，把通道变为3（即彩色图像），这样就达到了提高分辨率的目的。

## LapSRN
- CVPR2017
- 特征金字塔结构
