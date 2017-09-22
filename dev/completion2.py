from __future__ import print_function

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision
from   torchvision import datasets, transforms
from   torch.autograd import Variable
import torch.nn.functional as F
import shutil
import time
import matplotlib.pyplot as plt


cudnn.benchmark = True

class AdaptiveNorm2d(nn.Module):
    def __init__(self, nchannel, momentum = 0.05):
        super(AdaptiveNorm2d, self).__init__()
        self.nm = nn.BatchNorm2d(nchannel, momentum = momentum)
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w1 = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return self.w0.repeat(x.size()) * self.nm(x) +  self.w1.repeat(x.size()) * x
    
class CompletionNet2(nn.Module):
    def __init__(self):
        super(CompletionNet2, self).__init__()
        nf = 64
        alpha = 0.05
        self.convs = nn.Sequential(
            nn.Conv2d(5, nf/4, kernel_size = 5, stride = 1, padding = 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nf/4, nf, kernel_size = 5, stride = 2, padding = 2),
            AdaptiveNorm2d(nf, momentum=alpha),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nf, nf, kernel_size = 3, stride = 1, padding = 2),
            AdaptiveNorm2d(nf, momentum=alpha),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nf, nf*4, kernel_size = 5, stride = 2, padding = 1),
            AdaptiveNorm2d(nf * 4, momentum=alpha),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nf*4, nf * 4, kernel_size = 3, stride = 1, padding = 1),
            AdaptiveNorm2d(nf * 4, momentum=alpha),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, padding = 1),
            AdaptiveNorm2d(nf * 4, momentum=alpha),
            nn.LeakyReLU(0.1),

            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, dilation = 2, padding = 2),
            AdaptiveNorm2d(nf * 4, momentum=alpha),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, dilation = 4, padding = 4),
            AdaptiveNorm2d(nf * 4, momentum=alpha),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, dilation = 8, padding = 8),
            AdaptiveNorm2d(nf * 4, momentum=alpha),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, dilation = 16, padding = 16),
            AdaptiveNorm2d(nf * 4, momentum=alpha),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, dilation = 32, padding = 32),
            AdaptiveNorm2d(nf * 4, momentum=alpha),
            nn.LeakyReLU(0.1),

            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, padding = 1),
            AdaptiveNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf * 4, nf * 4, kernel_size = 3, stride = 1, padding = 1),
            AdaptiveNorm2d(nf * 4, momentum=alpha),
            nn.ReLU(),

            nn.ConvTranspose2d(nf * 4, nf , kernel_size = 4, stride = 2, padding = 1),
            AdaptiveNorm2d(nf , momentum=alpha),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nf, nf, kernel_size = 3, stride = 1, padding = 1),
            AdaptiveNorm2d(nf, momentum=alpha),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(nf, nf/4, kernel_size = 4, stride = 2, padding = 1),
            AdaptiveNorm2d(nf/4, momentum=alpha),
            nn.ReLU(),
            nn.Conv2d(nf/4, nf/4, kernel_size = 3, stride = 1, padding = 1),
            AdaptiveNorm2d(nf/4, momentum=alpha),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nf/4, 3, kernel_size = 3, stride = 1, padding = 1),
            )

    def forward(self, x, mask):
        return self.convs(torch.cat([x, mask], 1))

    
def identity_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.fill_(0)
        o, i, k1, k2 = m.weight.data.size()
        cx, cy = k1//2, k2//2
        nc = min(o,i)
        print(nc)
        for i in range(nc):
            m.weight.data[i,i,cx,cy] = 1
        m.bias.data.fill_(0)
        
        if m.stride[0] == 2:
            for i in range(nc):
                m.weight.data[i+nc,i,cx+1,cy] = 1
                m.weight.data[i+nc*2,i,cx,cy+1] = 1
                m.weight.data[i+nc*3,i,cx+1,cy+1] = 1
                
    
    elif classname.find('ConvTranspose2d') != -1:
        o, i, k1, k2 = m.weight.data.size()
        nc = min(o,i)
        cx, cy = k1//2-1, k2//2-1
        m.weight.data.fill_(0)
        for i in range(nc):
            m.weight.data[i,i,cx,cy] = 1
            m.weight.data[i+nc,i,cx+1,cy] = 1
            m.weight.data[i+nc*2,i,cx,cy+1] = 1
            m.weight.data[i+nc*3,i,cx+1,cy+1] = 1
        
        m.bias.data.fill_(0)
        
        
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)
