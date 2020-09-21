#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from matplotlib import pyplot as plt
import pandas as pd


# In[2]:


#加载数据集
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True,
download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False,
download=True, transform=transforms.ToTensor())

#划分数据集
n_data = 10000
temp = []
for i in range(n_data):
    temp.append(list(mnist_train[i][0]))
labels = []
for i in range(n_data):
    labels.append(mnist_train[i][1])
labels = torch.LongTensor(labels).view(-1,1)
features = temp[0][0].view(-1,784)
for i in range(1,n_data):
    features = torch.cat((features,temp[i][0].view(-1,784)), dim=0)


# In[3]:


#定义模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

#定义随机初始化模型参数的函数
W1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_inputs)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_outputs, num_hiddens)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
params = [W1, b1, W2, b2]

#定义linear层函数
def linear(X, w, b):
    return torch.matmul(X, w.t())+ b

#定义隐藏层激活函数
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

#定义分类模型
def net(X, W1, W2, b1, b2):
    X = X.view((-1, num_inputs))          
    linear_1 = linear(X, W1, b1)
    R = relu(linear_1)
    linear_2 = linear(R, W2, b2)
    return linear_2

#定义交叉熵损失函数，CrossEntropyLoss自带Softmax，所以分类模型中不用加Softmax层
loss = torch.nn.CrossEntropyLoss()

#定义随机梯度下降函数
def SGD(params, lr):
    for param in params:
        param.data -= lr * param.grad


# In[4]:


#定义k者交叉验证
def get_kfold_data(k, i, X, y):
    
    fold_size = X.shape[0] // k

    val_start = i * fold_size           
    if i != k-1:
        val_end = (i+1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = torch.cat( (X[:val_start], X[val_end:]), dim=0 )
        y_train = torch.cat( (y[:val_start], y[val_end:]), dim=0 )
    else:
        X_valid, y_valid = X[val_start:], y[val_start:]
        X_train = X[0:val_start]
        y_train = y[0:val_start]
    
    return X_train, y_train, X_valid, y_valid

#每一折的实验结果
train_loss_sum, valid_loss_sum = [],[]
train_acc_sum, valid_acc_sum = [],[]

def k_fold(k, X_train, y_train):
    
    for i in range(k):
        train_features, train_labels, test_features, test_labels = get_kfold_data(k, i, X_train, y_train)
        train_loss, valid_loss, train_acc, valid_acc = train(net, train_features, train_labels, test_features, test_labels)
        train_loss_sum.append(train_loss)
        valid_loss_sum.append(valid_loss)
        train_acc_sum.append(train_acc)
        valid_acc_sum.append(valid_acc)
        print('第%d折交叉验证结果：train_loss %f, valid_loss %f, train_acc %f, valid_acc %f' % 
              (i+1, train_loss, valid_loss, train_acc, valid_acc))
    print('\n')
    print('最终k折交叉验证结果：')
    print('average train loss %f, average train accuracy %f' % 
          (np.array(train_loss_sum).mean().item(), np.array(train_acc_sum).mean().item()))
    print('average valid loss %f, average valid accuracy %f' % 
          (np.array(valid_loss_sum).mean().item(), np.array(valid_acc_sum).mean().item()))

def train(net, X_train, y_train, X_valid, y_valid):
    
    #定义训练参数
    batch_size, num_epochs, lr = 128, 10, 0.01

    #定义随机初始化模型参数的函数
    W1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_inputs)), dtype=torch.float, requires_grad=True)
    b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
    W2 = torch.tensor(np.random.normal(0, 0.01, (num_outputs, num_hiddens)), dtype=torch.float, requires_grad=True)
    b2 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
    params = [W1, b1, W2, b2]
    
    #划分数据集
    dataset = torch.utils.data.TensorDataset(X_train,y_train)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            y_hat = net(X, W1, W2, b1, b2)
            l = loss(y_hat,y.view(-1)).sum() 
            l.backward()       #计算损失     
            SGD(params, lr)       #更新梯度
            W1.grad.data.zero_()
            b1.grad.data.zero_()
            W2.grad.data.zero_()
            b2.grad.data.zero_()
    loss_train = loss(net(X_train, W1, W2, b1, b2),y_train.view(-1)).mean().item() #训练集损失
    loss_test = loss(net(X_valid, W1, W2, b1, b2),y_valid.view(-1)).mean().item()    #测试集损失
    acc_train = (net(X_train, W1, W2, b1, b2).argmax(dim=1) == y_train.view(-1)).sum().item() / (y_train.shape[0]) #训练集准确率
    acc_test = (net(X_valid, W1, W2, b1, b2).argmax(dim=1) == y_valid.view(-1)).sum().item() / (y_valid.shape[0])  #测试集准确率
    #最后训练时的误差会是本次代码运行的最优误差，所以只需要返回最后训练的误差即可
    return loss_train, loss_test, acc_train, acc_test

k_fold(10, features, labels)


# In[5]:


data = {"train_loss":train_loss_sum,"valid_loss":valid_loss_sum,"train_acc":train_acc_sum,"valid_acc":valid_acc_sum}
pd.DataFrame(data,index=range(1,11))


# In[ ]:




