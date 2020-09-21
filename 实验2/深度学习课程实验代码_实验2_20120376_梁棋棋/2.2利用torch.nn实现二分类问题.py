#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
num_inputs, num_outputs, num_hiddens = 200, 1, 256

n_data = torch.ones((n_train+n_test), num_inputs) 
x1 = torch.normal(2 * n_data, 1)      #正例特征
y1 = torch.zeros((n_train+n_test),1)  #正例标签
x2 = torch.normal(-2 * n_data, 1)     #负例特征
y2 = torch.ones((n_train+n_test),1)   #负例标签

x = torch.cat((x1, x2), 0).type(torch.FloatTensor)
y = torch.cat((y1, y2), 0).type(torch.FloatTensor)

#分出训练集样本、测试集样本、训练集标签、测试集标签
train_features = torch.cat((x1[:n_train, :],x2[:n_train, :]), 0)
test_features = torch.cat((x1[n_train:, :],x2[n_train:, :]), 0)
train_labels = torch.cat((y1[:n_train, :],y2[:n_train, :]), 0)
test_labels = torch.cat((y1[n_train:, :],y2[n_train:, :]), 0)


# In[5]:


class Binary_Classification(torch.nn.Module):
    def __init__(self):
        super(Binary_Classification, self).__init__()
        self.hidden = torch.nn.Linear(num_inputs, num_hiddens) #定义隐藏层函数
        self.relu = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.output = torch.nn.Linear(num_hiddens, num_outputs)#定义输出层函数
        self.sigmoid = torch.nn.Sigmoid()                      #定义输出层激活函数，此处定义Sigmoid了，损失函数可以只用BCELoss

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
        
net = Binary_Classification()

#初始化模型参数
init.normal_(net.hidden.weight, mean=0, std=0.01)
init.normal_(net.output.weight, mean=0, std=0.01)
init.constant_(net.hidden.bias, val=0)
init.constant_(net.output.bias, val=0)


# In[6]:


#定义训练参数
batch_size, num_epochs, lr = 128, 100, 0.003

#定义损失函数和优化器
loss = torch.nn.BCELoss()  # BCEWithLogitsLoss = BCELoss + Sigmoid
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
        optimizer.step()   #更新梯度
    loss_train.append(loss(net(train_features),train_labels).mean().item()) #训练集损失
    loss_test.append(loss(net(test_features),test_labels).mean().item())    #测试集损失
    print('epoch %d, loss_train %f, loss_test %f '% (epoch+1, loss_train[epoch], loss_test[epoch]))


# In[7]:


plt.figure(figsize=(10,7))
plt.plot(loss_train,label='train',color='royalblue')
plt.plot(loss_test,label='test',linestyle=':', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.show()


# In[ ]:




