import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
import sys
path = os.path.abspath(os.path.dirname(sys.argv[0]))
import torch.nn.functional as F

picture_name = "head"

savepath=r'features_map'
if not os.path.exists(savepath):
    os.mkdir(savepath)

def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
        img=img.astype(np.uint8)  #转成unit8
        # img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
        # img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        plt.show()
        print("{}/{}".format(i,width*height))
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    # plt.margins(0,0)
    #fig.savefig(savename, format='png', transparent=True, dpi=300, pad_inches = 0)

    fig.savefig(savename,bbox_inches='tight')
    # fig.clf()
    plt.close()
    print("time:{}".format(time.time()-tic))
 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.weight = nn.Parameter(torch.randn(16, 1, 5, 5))  # 自定义的权值
        self.bias = nn.Parameter(torch.randn(16))    # 自定义的偏置
 
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = F.conv2d(x, self.weight, self.bias, stride=1, padding=0)
        return out

class ft_net(nn.Module):
    def __init__(self):
        super(ft_net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3,out_channels=4,kernel_size=3,stride=1,padding=1)
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        if True: # draw features or not
            #x = self.conv_1(x) 
            x0 = x[:, 0]
            x1 = x[:, 1]
            x2 = x[:, 2]
            x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
            x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

            x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
            x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

            x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
            x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

            x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
            x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
            x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

            x = torch.cat([x0, x1, x2], dim=1)
            #x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
            draw_features(1,1,x0_v.cpu().numpy(),"{}/{}_conv1.png".format(savepath,picture_name))
        return x
 
model=ft_net().cuda()

model.eval()
img=cv2.imread(path + '\\{}.png'.format(picture_name))
#img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print(img.shape)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img=transform(img).cuda()
img=img.unsqueeze(0)
 
with torch.no_grad():
    start=time.time()
    out=model(img)
    print("total time:{}".format(time.time()-start))
    result=out.cpu().numpy()
    ind=np.argsort(result,axis=1)
    print("done")