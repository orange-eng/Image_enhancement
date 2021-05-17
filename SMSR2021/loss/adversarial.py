import utility
from model import common
from loss import discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k     # k value for adversarial loss
        self.discriminator = discriminator.Discriminator(args, gan_type)    # gan_type='GAN'
        if gan_type != 'WGAN_GP':
            self.optimizer = utility.make_optimizer(args, self.discriminator)
        else:
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=1e-5
            )
        self.scheduler = utility.make_scheduler(args, self.optimizer)

    def forward(self, fake, real):
        fake_detach = fake.detach()                     ## detach()将variable参数从网络中隔离开，不参与参数更新

        self.loss = 0
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()                  #直接把模型的参数梯度设成0
            d_fake = self.discriminator(fake_detach)    # 得到虚假图片的判断结果
            d_real = self.discriminator(real)           # 得到真实图片的判别结果
            if self.gan_type == 'GAN':
                label_fake = torch.zeros_like(d_fake)   # 生成一个值全为0的、维度与输入尺寸相同的矩阵
                label_real = torch.ones_like(d_real)    # 生成一个值全为1的、维度与输入尺寸相同的矩阵
                loss_d \
                    = F.binary_cross_entropy_with_logits(d_fake, label_fake) \
                    + F.binary_cross_entropy_with_logits(d_real, label_real)    # 计算多分类问题的交叉熵loss  
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            # Discriminator update
            self.loss += loss_d.item()
            loss_d.backward()
            self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k                     # a = a/b， 每次缩小k倍

        d_fake_for_g = self.discriminator(fake)     # 判别器对虚假图片的输出
        if self.gan_type == 'GAN':
            loss_g = F.binary_cross_entropy_with_logits(    # 生成器的损失函数（交叉熵）
                d_fake_for_g, label_real
            )
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_for_g.mean()

        # Generator loss
        return loss_g
    
    def state_dict(self, *args, **kwargs):
        state_discriminator = self.discriminator.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)
               
# Some references
# https://github.com/kuc2477/pytorch-wgan-gp/blob/master/model.py
# OR
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py
