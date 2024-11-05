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
import matplotlib.animation as animation
from IPython.display import HTML

from generator import Generator
from discriminator import Discriminator

bt_sz=64
im_sz=64
nc = 3
nz = 100
ngf = 64
ndf = 64
ngpu=1
lr = 0.0002
beta1 = 0.5
num_ep=5
transform=trans.Compose([trans.Resize(im_sz),trans.ToTensor(),trans.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
dataset=ds.CIFAR10(root='./data',download=True,transform=transform)
dataload=ud.DataLoader(dataset,batch_size=bt_sz,shuffle=True,num_workers=2)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")

# 
#  rl_bt=next(iter(dataload))
#  plt.figure(figsize=(8,8))
#  plt.axis("off")
#  plt.title("Training Images")
#  plt.imshow(np.transpose(vut.make_grid(rl_bt[0].to(device)[:69],padding=2,normalize=True).cpu(),(1,2,0)))
#  plt.show()

if __name__=='__main__':
 def weight_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)

 netD=Discriminator(ngpu,nc,ndf).to(device)
 netG=Generator(ngpu,nz,ngf,nc).to(device)

 if(device.type=='cuda')and(ngpu>1):
  netD=nn.DataParallel(netD,list(range(ngpu)))
  netG=nn.DataParallel(netG,list(range(ngpu)))


 netD.apply(weight_init)
 netG.apply(weight_init)

 criterion=nn.BCELoss()

 fx_no=torch.randn(64,nz,1,1,device=device)

 re_lb=1
 fk_lb=0

 opti_d=op.Adam(netD.parameters(),lr=lr,betas=(beta1,0.999))
 opti_g=op.Adam(netG.parameters(),lr=lr,betas=(beta1,0.999))


 img_list = []
 G_losses = []
 D_losses = []
 iters = 0

 print("Starting Training Loop...")

 for epoch in range(num_ep):
    # For each batch in the dataloader
    for i, data in enumerate(dataload, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), re_lb, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fk_lb)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        opti_d.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(re_lb)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        opti_g.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_ep, i, len(dataload),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_ep-1) and (i == len(dataload)-1)):
            with torch.no_grad():
                fake = netG(fx_no).detach().cpu()
            img_list.append(vut.make_grid(fake, padding=2, normalize=True))

        iters += 1

 plt.figure(figsize=(10,5))
 plt.title("Generator and Discriminator Loss During Training")
 plt.plot(G_losses,label="G")
 plt.plot(D_losses,label="D")
 plt.xlabel("iterations")
 plt.ylabel("Loss")
 plt.legend()
 plt.show()

 fig = plt.figure(figsize=(8,8))
 plt.axis("off")
 ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
 ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
 HTML(ani.to_jshtml())

 real_batch = next(iter(dataload))

 plt.figure(figsize=(15,15))
 plt.subplot(1,2,1)
 plt.axis("off")
 plt.title("Real Images")
 plt.imshow(np.transpose(vut.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

 plt.subplot(1,2,2)
 plt.axis("off")
 plt.title("Fake Images")
 plt.imshow(np.transpose(img_list[-1],(1,2,0)))
 plt.show()