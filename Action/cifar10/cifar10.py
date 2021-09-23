from d2l.torch import accuracy
from torchvision import transforms
import numpy as np
import torch.utils.data as Data
import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch import optim

batch_size = 128
cifar10_mean = (0.5,0.5,0.5)
cifar10_std = (0.5,0.5,0.5)
data_path = r'learnNotes\Action\cifar10\data'
train_transform = transforms.Compose([transforms.Pad(4),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32),
                                      transforms.ToTensor(),
                                      transforms.Normalize(cifar10_mean,cifar10_std) ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(cifar10_mean,
                                                          cifar10_std)])
train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=train_transform, download=False)
test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=test_transform, download=False )
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
class Convnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d(16 ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))#batch_size*64*4*4
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d( 32 ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))#batch_size*64*4*4
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.BatchNorm2d( 64 ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))#batch_size*64*4*4

        self.out = nn.Sequential(nn.Linear(64 * 4* 4,10))#输出层

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        out = self.out(x)
        return out

# for X,y in train_loader:
#     print(X.shape,y)
#     break


model = Convnet()#实例


def acc( y_hat ,y ):
    if len( y_hat.shape ) >1 and y_hat.shape[1]>1:
        y_hat = torch.argmax( y_hat , dim=1 )
    count = y_hat.type( y.dtype ) == y
    return count.type( y.dtype ).sum()
def evaluate_acc( net , data_iter ):
    if isinstance( net , nn.Module ):
        model.eval()
    acc_numbel = 0
    all_numbel = 0
    for X,y in data_iter:
        X,y = X.to( try_gpu()) , y.to( try_gpu() )
        acc_numbel += accuracy( net(X) , y )
        all_numbel+=y.numel()
    return acc_numbel/all_numbel
# print( evaluate_acc( model , train_loader ) )

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device( f'cuda:{i}')
    return torch.device( 'cpu' )

def train( device ):

    # model.train()
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    model.apply(init_weights)
    print('training on', device)
    model.to( device )
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimser = torch.optim.Adam(model.parameters(), lr=learning_rate )
    for epoch in range(10):
        print('current epoch=%d' % epoch)
        for i,(images,label)in enumerate(train_loader):
            # images=images.view(images.size(0),-1)
            images,label = images.to(device),label.to(device)
            optimser.zero_grad()
            outputs=model(images)
            # print( outputs.shape , label.shape )
            loss=criterion( outputs,label )
            loss.backward()
            optimser.step()
            if i%batch_size==0:
                print('crrent loss=%.5f'%loss.item())
        with torch.no_grad():        
            print( f'当前测试集准确率为：{evaluate_acc( model , test_loader )}' )
    print('完成训练')

    # model.eval()
    # correct_num=0
    # total_num=0
    # with torch.no_grad():
    #     for images,labels in test_loader:
    #         # images=images.view(images.size(0),-1)
    #         outputs=model(images)
    #         _,pred=torch.max(outputs.data,dim=1)
    #         total_num+=labels.size(0)
    #         correct_num+=(pred==labels).sum()
    # print('当前正确率为：'(100*  float( correct_num) / float(total_num) ))

train( try_gpu() )