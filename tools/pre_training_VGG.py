'''
import torch
import torchvision

model_vgg16 = torchvision.models.vgg16(pretrained=True)
#print(model_vgg16)
num_fc = model_vgg16.classifier[6].in_features
#print(num_fc)
model_vgg16.classifier[6] = torch.nn.Linear(num_fc,2)
#print(model_vgg16)

# 对每一个权重，固定参数
for param in model_vgg16.parameters():
    param.requires_grad = False

# 不需要 固定最后一层
for param in model_vgg16.classifier[6].parameters():
    param.requires_grad = True

# for child in model_vgg16.children():
#     print(child)
#     for param in child.parameters():
#         print(param)
print(model_vgg16.features[30])
'''
import math
import torch
from torch import nn
from torchvision import models
from torchsummary import summary
#model = models.vgg19(pretrained=False)
#model_vgg16 = models.vgg16(pretrained=True)
#model = models.resnet18(pretrained=True)
#model = models.inception_v3(pretrained=True)
model_mobile = models.mobilenet_v2(pretrained=True)

model_mobile.eval()
#model_vgg16.eval()

#summary(model_vgg16,(3,224,224))

print(model_mobile.features[2])

# for child in model_mobile.features:
#     print(child)
'''
print('model.features[9]',model.features[0]) #根据观看model形状表示而得，该层为MaxPool2d层

# 给定输入，可以得到相应的输出结果
input = torch.randn(1, 3, 256, 278)
print(input)
# print(len(input))
pre = model.features[0](input)  # 12,64,256,278
print(pre.shape)    
pre = model.features[1](pre)
pre = model.features[2](pre)
pre = model.features[3](pre)
pre = model.features[4](pre)    #12,64,128,139
#pre = model.features[2](pre)

#print(pre)
print(pre.shape)
'''