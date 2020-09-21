#!/usr/bin/env python
# coding: utf-8

# In[3]:


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import datetime

path = "./车辆分类数据集/bus/bus001.jpg"
train_features = torch.Tensor(np.array(Image.open(path).resize((100,100),Image.ANTIALIAS))/255).view(1, 100, 100, 3)
test_features = torch.Tensor(np.array(Image.open(path).resize((100,100),Image.ANTIALIAS))/255).view(1, 100, 100, 3)

# 处理客车数据，总共218张，前153张为训练集，后65张为测试集
pic_num = 0
for filename in os.listdir("./车辆分类数据集/bus/"): 
    pic_num = pic_num + 1
    path = "./车辆分类数据集/bus/"+ filename
    img = torch.Tensor(np.array(Image.open(path).resize((100,100),Image.ANTIALIAS))/255).view(1, 100, 100, 3)
    if pic_num <= 153:
        train_features = torch.cat((train_features,img), dim=0)
    else:
        test_features = torch.cat((test_features,img), dim=0)

# 处理汽车数据，总共779张，前545张为训练集，后234张为测试集
pic_num = 0
for filename in os.listdir("./车辆分类数据集/car/"): 
    pic_num = pic_num + 1
    path = "./车辆分类数据集/car/"+ filename
    img = torch.Tensor(np.array(Image.open(path).resize((100,100),Image.ANTIALIAS))/255).view(1, 100, 100, 3)
    if pic_num <= 545:
        train_features = torch.cat((train_features,img), dim=0)
    else:
        test_features = torch.cat((test_features,img), dim=0)

# 处理货车数据，总共360张，前252张为训练集，后108张为测试集
pic_num = 0
for filename in os.listdir("./车辆分类数据集/truck/"): 
    pic_num = pic_num + 1
    path = "./车辆分类数据集/truck/"+ filename
    img = torch.Tensor(np.array(Image.open(path).resize((100,100),Image.ANTIALIAS))/255).view(1, 100, 100, 3)
    if pic_num <= 252:
        train_features = torch.cat((train_features,img), dim=0)
    else:
        test_features = torch.cat((test_features,img), dim=0)

train_features = train_features.permute(0,3,2,1)
test_features = test_features.permute(0,3,2,1)
print(train_features.shape)
print(test_features.shape)

# =================== 训练集标签 =================== #
train_labels = torch.zeros(154).long()
train_labels = torch.cat((train_labels,torch.ones(545).long()), dim=0)
train_labels = torch.cat((train_labels,torch.ones(252).long()+1), dim=0)
print(train_labels.shape)

# =================== 测试集标签 =================== #
test_labels = torch.zeros(66).long()
test_labels = torch.cat((test_labels,torch.ones(234).long()), dim=0)
test_labels = torch.cat((test_labels,torch.ones(108).long()+1), dim=0)
print(test_labels.shape)


# In[6]:


#自定义卷积层

def corr2d(X, K):
    """
    X: 输入, shape(batch_size, H, W)
    K: 卷积核,shape(k_h, k_w)
    """
    batch_size, H, W = X.shape
    #print("\ncorr2d:")
    #print(X.shape)
    #print(batch_size, H, W)
    k_h, k_w = K.shape
    #初始化结果矩阵
    Y = torch.zeros((batch_size, H-k_h+1, W-k_w+1))
    for i in range(Y.shape[1]):
        for j in range(Y.shape[2]):
            Y[:,i,j] = (X[: , i:i+k_h , j:j+k_w] * K).sum()
    return Y

def corr2d_multi_in(X, K):
    """
    X: 输入, shape(batch_size, C_in, H, W)
    K: 卷积核, shape(C_in, k_h, k_w)
    return: 输出, shape(batch_size, H_out, W_out)
    """
    #print("\ncorr2d_multi_in")
    #print(X.shape) #torch.Size([32, 3, 100, 100])
    #print(K.shape) #torch.Size([32, 3, 3])
    res = corr2d(X[: , 0 , : , : ], K[0, : , : ])
    for i in range(1, X.shape[1]):
        res += corr2d(X[: , i , : , : ], K[i, : , : ])
    return res

def corr2d_multi_in_out(X, K):
    """
    X: 输入, shape(batch_size, C_in, H, W)
    K: 卷积核, shape(C_out, C_in, h, w)
    return: 输出, shape(batch_size, C_out, H_out, W_out)
    """
    #print("\ncorr2d_multi_in_out")
    #print(X.shape)  #torch.Size([32, 3, 100, 100])
    #print(K.shape)  #torch.Size([32, 3, 3, 3])
    #对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K] , dim=1)

class MyConv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MyConv2D, self).__init__()
        if isinstance(kernel_size,int):
            kernel_size = (kernel_size, kernel_size)
        self.weight = torch.nn.Parameter(torch.randn((out_channels, in_channels)+kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(out_channels,1,1))
 
    def forward(self, x):
        """
        x: 输入图片, shape(batch_size, C_in, H, W)
        """
        #print(x.shape) #torch.Size([32, 3, 100, 100])
        #print(self.weight.shape) #torch.Size([32, 3, 3, 3])
        return corr2d_multi_in_out(x, self.weight) + self.bias
    
class MyConvModule(torch.nn.Module):
    def __init__(self):
        super(MyConvModule, self).__init__()
        self.conv = nn.Sequential(
            MyConv2D(in_channels=3, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)           
        )
        self.fc = nn.Linear(32, num_classes) #[32,3]
 
    def forward(self, X):
        out = self.conv(X)
        #print(out.shape)  #torch.Size([32, 32, 98, 98])
        out = nn.functional.avg_pool2d(out, 98)
        #print(out.shape) #torch.Size([32, 32, 1, 1])
        out = out.squeeze()
        #print(out.shape) #torch.Size([32, 32])
        out = self.fc(out)
        return out


# In[7]:


#训练函数
def train_epoch(net, data_loader):
    
    net.train()
    train_batch_num = len(data_loader)
    total_loss = 0
    correct = 0
    sample_num = 0
    
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = net(data)
        print(output)
        loss = criterion(output, target)
        print(loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        prediction = torch.argmax(output, 1)
        correct += (prediction == target).sum().item()
        sample_num += len(prediction)
        print(batch_idx)
        print(total_loss,correct/sample_num)
    
    loss = total_loss / train_batch_num
    acc = correct / sample_num
    return loss, acc

#测试函数
def test_epoch(net, data_loader):
    
    net.eval()
    test_batch_num = len(data_loader)
    total_loss = 0
    correct = 0
    sample_num = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            prediction = torch.argmax(output, 1)
            correct += (prediction == target).sum().item()
            sample_num += len(prediction)
            print(batch_idx)
            print(total_loss,correct/sample_num)
    
    loss = total_loss / train_batch_num
    acc = correct / sample_num
    return loss, acc


# In[ ]:


num_classes = 3
num_epoch = 1
lr = 0.001
batch_size = 32

#划分数据集
train_dataset = torch.utils.data.TensorDataset(train_features,train_labels)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
test_dataset = torch.utils.data.TensorDataset(test_features,test_labels)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True, num_workers=0)

net = MyConvModule()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

train_loss_sum = []
train_acc_sum = []
test_loss_sum = []
test_acc_sum = []

for epoch in range(num_epoch):
    
    train_loss, train_acc = train_epoch(net, data_loader=train_iter)
    test_loss, test_acc = test_epoch(net, data_loader=test_iter)
    
    train_loss_sum.append(train_loss)
    train_acc_sum.append(train_acc)
    test_loss_sum.append(test_loss)
    test_acc_sum.append(test_acc)
    
    print('epoch %d, train_loss %f, test_loss %f, train_acc %f, test_acc %f' % 
          (epoch+1, train_loss, test_loss, train_acc, test_acc))


# In[ ]:




