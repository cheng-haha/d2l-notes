'''
Author: your name
Date: 2021-09-02 10:46:37
LastEditTime: 2021-09-06 15:08:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \d2l-zh\learnNotes\chapter_muktilayer\multilayer_zeros_py.py
'''
import torch
from torch import nn
from d2l import torch as d2l
from torch.nn.modules.loss import CrossEntropyLoss
nn.ReLU
# batch_size = 18 
# train_iter , test_iter = d2l.load_data_fashion_mnist(batch_size)
ones = torch.Tensor( [ 1,0,1,0])
zeros = torch.zeros( 4 ) 
# zeros.reshape
CrossEntropyLoss
print(  torch.max( ones , zeros ) )
d2l.linreg
d2l.evaluate_accuracy
d2l.sgd
d2l.Animator
d2l.set_axes
d2l.load_data_fashion_mnist
d2l.train_ch3
d2l.load_array
torch.optim.Adam
nn.Conv2d