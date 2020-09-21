#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torch.utils import data
from matplotlib import pyplot as plt
import pandas as pd


# In[2]:


#训练集7000，测试集3000（加起来为数据集大小10000）
n_train, n_test  = 7000, 3000

#定义模型参数
num_inputs = 500

#高维线性函数：y = 0.028+Σ0.0056x+ϵ   true_w[500,1]
true_w, true_b = torch.ones(num_inputs,1)*0.0056, 0.028

features = torch.randn((n_train+n_test),num_inputs)
labels = torch.matmul(features, true_w) + true_b  #y = 0.028+Σ0.0056x  [7000,500]*[500,1]=[7000,1]
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),dtype=torch.float) #加上噪声项ϵ


# In[3]:


#定义随机初始化模型参数的函数
w = torch.randn((num_inputs,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)
params = [w, b]

#定义回归模型
def net(X,w,b):
    return torch.mm(X,w) + b

#定义均方误差（回归函数损失函数可使用均方误差）
def squared_loss(y_hat,y):
    return ((y_hat-y.view(y_hat.size())) ** 2) / 2

#定义随机梯度下降函数
def SGD(params, lr):
    for param in params:
        param.data -= lr * param.grad


# In[4]:


#定义k折交叉验证
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
        print('第%d折交叉验证结果：train_loss %f, valid_loss %f' % (i+1, train_loss, valid_loss))
    print('\n')
    print('最终k折交叉验证结果：')
    print('average train loss:{:f} '.format(np.array(train_loss_sum).mean().item()))
    print('average valid loss:{:f} '.format(np.array(valid_loss_sum).mean().item()))
    #print('average train loss:{:f}, average train accuracy:{:f} %'.format(train_loss_sum/k, train_acc_sum/k))
    #print('average valid loss:{:f}, average valid accuracy:{:f} %'.format(valid_loss_sum/k, valid_acc_sum/k))

def train(net, X_train, y_train, X_valid, y_valid):
    
    #定义训练参数
    batch_size, num_epochs, lr = 128, 20, 0.003

    #划分数据集
    dataset = torch.utils.data.TensorDataset(X_train,y_train)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            y_hat = net(X,w,b)
            loss = squared_loss(y_hat,y).sum() 
            loss.backward()       #计算损失     
            SGD(params, lr)    #更新梯度
            w.grad.data.zero_()
            b.grad.data.zero_()
    loss_train = squared_loss(net(X_train,w,b),y_train).mean().item() #训练集损失
    loss_test = squared_loss(net(X_valid,w,b),y_valid).mean().item()    #测试集损失
    #最后训练时的误差会是本次代码运行的最优误差，所以只需要返回最后训练的误差即可(回归问题不涉及准确率)
    return loss_train, loss_test, 0, 0

k_fold(10, features, labels)


# In[13]:


data = {"train_loss":train_loss_sum,"valid_loss":valid_loss_sum}
pd.DataFrame(data,index=range(1,11))


# In[16]:


print(w.mean().item(),b.item())


# In[ ]:




