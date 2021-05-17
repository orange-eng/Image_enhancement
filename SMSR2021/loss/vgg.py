from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

class VGG(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])  #提取VGG 第8层的图像特征
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35]) #提取VGG 35层的图像特征

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False
        #requires_grad 说明当前量是否需要在计算中保留对应的梯度信息
        # 只要某一个输入需要相关梯度值，则输出也需要保存相关梯度信息，这样就保证了这个输入的梯度回传。
        # 而反之，若所有的输入都不需要保存梯度，那么输出的requires_grad会自动设置为False。既然没有了相关的梯度值，自然进行反向传播时会将这部分子图从计算中剔除。
    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)        # 经过偏移之后
            x = self.vgg(x)             # 从VGG中提取特征图
            return x
            
        vgg_sr = _forward(sr)   #获取特征图
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())  # detach()将variable参数从网络中隔离开，不参与参数更新

        loss = F.mse_loss(vgg_sr, vgg_hr)   #损失函数为MSE

        return loss
