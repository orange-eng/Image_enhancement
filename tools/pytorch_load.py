import torch
import torch.nn as nn

class Net_old(nn.Module):
    def __init__(self):
        super(Net_old, self).__init__()
        self.nets = nn.Sequential(
            torch.nn.Conv2d(1, 2, 3),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(2, 1, 3),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(1, 1, 3)
        )
    def forward(self, x):
        return self.nets(x)

class Net_new(nn.Module):
    def __init__(self):
        super(Net_old, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 3)
        self.r1 = torch.nn.ReLU(True)
        self.conv2 = torch.nn.Conv2d(2, 1, 3)
        self.r2 = torch.nn.ReLU(True)
        self.conv3 = torch.nn.Conv2d(1, 1, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.r1(x)
        x = self.conv2(x)
        x = self.r2(x)
        x = self.conv3(x)
        return x

network = Net_old()
torch.save(network.state_dict(), 't.pth') #存储模型的参数到t.pth文件中
pretrained_net = torch.load('t.pth')    #pretrained_net文件是一个OrderedDict类型文件，存放各种参数
#print(pretrained_net)
# for key, v in enumerate(pretrained_net):
#     print(key, v)
i = 0
for name, module in network.named_modules():
    i = i + 1
    print(i,module)
    for p_name, p_tensor in module.named_parameters():
        print(p_name)
# for key, v in enumerate(pretrained_net):
#     print(key,v)