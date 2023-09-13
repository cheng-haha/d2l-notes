'''
Author: your name
Date: 2021-10-02 11:20:24
LastEditTime: 2021-10-02 11:27:56
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \d2l-zh\learnNotes\chapter_recurrent-neural-networks\RNNScratch.py
'''
import math
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
import torch
from typing import Tuple,List,Dict,Callable
from torch import Tensor

#批次为32 ， 时间步为35
batch_size , num_steps = 32 , 35
train_iter , vocab =  d2l.load_data_time_machine( batch_size , num_steps )

# ## 8.5.1 独热编码


x = torch.randint( 0 , 3 , ( 2,5 ))
x


x_onehot =  F.one_hot( x.T  ,28)#注意如果没有指定元素个数1的话，会自动匹配多少个元素，比如说这里有0 1 2 这三个元素，那就需要3个维度，最后得到一个2*3的形状
x_onehot , x_onehot.shape #维度为：批次 * 长度 * 元素个数



#我们需要得到一个 （时间步数, 批量大小, 词汇表大小）的输出
x_onehot =  F.one_hot( x.T  )
print( x_onehot )
for x in x_onehot:
    print( x )



def get_params( vocab_size ,num_hiddens , device ):
    num_inputs = num_outputs  = vocab_size
    def normal( shape ):
        return torch.randn( size=shape , device = device) * 0.01
    
    #隐藏层参数
    #输入与权值参数相乘
    w_xh = normal( (num_inputs , num_hiddens) )
    #上次的隐藏变量与权值参数相乘
    w_hh = normal( (num_hiddens , num_hiddens) )
    #偏置
    b_hh = torch.zeros( num_hiddens , device = device )

    #输出层参数
    w_hq = normal( (num_hiddens , num_outputs) )
    b_hq = torch.zeros( num_outputs , device= device  )

    #将参数组装成一个列表
    params = [ w_xh , w_hh , b_hh , w_hq , b_hq ]
    #保留梯度参数
    for param in params:
        param.requires_grad_( True )
    return params
    



#保留隐藏状态，返回一个元组,函数的返回是一个张量，张量全用0填充，形状为（批量大小, 隐藏单元数）。
#在后面的章节中将会遇到隐藏状态包含多个变量的情况，而使用元组可以处理地更容易些。
def init_rnn_state( batch_size , num_hiddens  , device ):
    return (torch.zeros( (batch_size , num_hiddens) , device = device ) , )



def rnn( inputs ,  state , params ):
    # `inputs`的形状：(`时间步数量`，`批量大小`，`词表大小`)，逐时间步更新参数
    w_xh,  w_hh , b_h , w_hq , b_q = params
    h, = state
    outputs = []
    # X 的形状为( 批量大小 ， 词表大小)
    for X in inputs:
        H = torch.tanh( torch.mm( X , w_xh ) + torch.mm( h , w_hh ) + b_h )
        Y = torch.mm( H , w_hq ) + b_q
        outputs.append( Y )
    #以行堆叠数据,返回输出序列 和当前隐藏状态
    return torch.cat( outputs , dim=0 ) , ( H , )



#使用一个类去包装当前函数
class RNNModelScratch:
    def __init__( self , vocab_size,  num_hiddens , device , get_params , init_state , forward_fn ):
        #得到词表大小和隐藏状态的张量大小
        self.vocab_size ,  self.num_hiddens = vocab_size , num_hiddens
        self.params = get_params( vocab_size , num_hiddens , device )
        self.forward_fn = forward_fn
    #这里我们使用__call__魔法方法将类可以当做函数进行使用，类似于forward,最后返回的是一个集成了输出和隐藏状态的元祖
    def __call__(self, X , state ) :
        X = F.one_hot( X.T , self.vocab_size ).type( torch.float32 )
        return self.forward_fn( X , state , self.params )
    
    def begin_state( self , batch_size , device ):
        return init_rnn_state( batch_size , self.num_hiddens , device )
        



X = torch.arange(10).reshape((2, 5))
F.one_hot(X.T, 28).shape , X



num_hiddens = 512 
net = RNNModelScratch( len( vocab ) , num_hiddens , d2l.try_gpu() , 
                      get_params , init_rnn_state , rnn )
state = net.begin_state( X.shape[0] , d2l.try_gpu() )
Y , new_state = net( X.to( d2l.try_gpu() ) , state )

