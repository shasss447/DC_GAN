from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as op
import torch.utils.data as ud
import torchvision.datasets as ds
import torchvision.transforms as trans
import torchvision.utils as vut
from torch.autograd import Variable

from generator import Generator
from discriminator import Discriminator

bt_sz=64
im_sz=64
nc = 3
nz = 100
ngf = 64
ndf = 64
ngpu=1

transform=trans.Compose([trans.Resize(im_sz),trans.ToTensor(),trans.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
dataset=ds.CIFAR10(root='./data',download=True,transform=transform)
dataload=ud.DataLoader(dataset,batch_size=bt_sz,shuffle=True,num_workers=2)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")

# if __name__=='__main__':
#  rl_bt=next(iter(dataload))
#  plt.figure(figsize=(8,8))
#  plt.axis("off")
#  plt.title("Training Images")
#  plt.imshow(np.transpose(vut.make_grid(rl_bt[0].to(device)[:69],padding=2,normalize=True).cpu(),(1,2,0)))
#  plt.show()


def weight_init(m):
 classname = m.__class__.__name__
 if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
 elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)

netD=Discriminator(ngpu,nc,ndf).to(device)

if(device.type=='cuda')and(ngpu>1):
 netD=nn.DataParallel(netD,list(range(ngpu)))

netD.apply(weight_init)
print(netD)