#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torch.utils import data
from torch.nn import init
from matplotlib import pyplot as plt
import torch.optim as optim


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

#分出训练集样本、测试集样本、训练集标签、测试集标签
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train, :], labels[n_train:, :]

print(train_features.shape)
print(test_features.shape)
print(train_labels.shape)
print(test_labels.shape)


# In[5]:


class LinearNet(torch.nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = torch.nn.Linear(n_feature, 1)
        
# forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y

#初始化模型参数
net = LinearNet(num_inputs) #样本维度
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0) #也可以直接修改bias的data：net[0].bias.data.fill_(0)


# In[7]:


#定义训练参数
batch_size, num_epochs, lr = 128, 100, 0.003

#定义损失函数和优化器
loss = torch.nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr)

#划分数据集
dataset = torch.utils.data.TensorDataset(train_features,train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

#训练模型
loss_train = []
loss_test = []
for epoch in range(num_epochs):
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y).sum() 
        optimizer.zero_grad()
        l.backward()       #计算损失     
        optimizer.step() #更新梯度
    loss_train.append(loss(net(train_features),train_labels).mean().item()) #训练集损失
    loss_test.append(loss(net(test_features),test_labels).mean().item())    #测试集损失
    print('epoch %d, loss_train %f, loss_test %f '% (epoch+1, loss_train[epoch], loss_test[epoch]))


# In[8]:


plt.figure(figsize=(10,7))
plt.plot(loss_train,label='train',color='royalblue')
plt.plot(loss_test,label='test',linestyle=':', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.show()


# In[12]:


print(net.linear.weight.mean())
print(net.linear.bias)


# In[ ]:




