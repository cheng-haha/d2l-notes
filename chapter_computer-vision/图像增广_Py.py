import torch
import torchvision
from torch import nn 
from d2l import torch as d2l

# d2l.set_figsize()#设置画布大小
# img = d2l.Image.open('cat.JPG')
# d2l.plt.imshow(img)
a = torch.randn( size=( 1000 , 1000 ) , device=d2l.try_gpu() )
torch.mm( a , a )