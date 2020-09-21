#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torch.utils import data
from torch.nn import init
from matplotlib import pyplot as plt
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# In[2]:


#加载数据集
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True,
download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False,
download=True, transform=transforms.ToTensor())

#定义模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

#定义计算测试集损失函数
def evaluate_loss(data_iter, net):
    l, n = 0.0, 0
    for X, y in data_iter:
        X = X.view((-1, num_inputs)) 
        l += loss(net(X),y).sum().item()
        n += y.shape[0]
    return l / n

#定义计算测试集准确率函数
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        X = X.view((-1, num_inputs)) 
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

#定义训练参数
batch_size, num_epochs, lr = 256, 30, 0.01

#划分数据集
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,num_workers=0)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,num_workers=0)


# In[3]:


# ========== Relu激活函数 ==========

class Relu_Classification(torch.nn.Module):
    def __init__(self):
        super(Relu_Classification, self).__init__()
        self.hidden = torch.nn.Linear(num_inputs, num_hiddens) #定义隐藏层函数
        self.relu = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.output = torch.nn.Linear(num_hiddens, num_outputs)#定义输出层函数

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x
        
net = Relu_Classification()

#初始化模型参数
init.normal_(net.hidden.weight, mean=0, std=0.01)
init.normal_(net.output.weight, mean=0, std=0.01)
init.constant_(net.hidden.bias, val=0)
init.constant_(net.output.bias, val=0)

#定义损失函数和优化器
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr)

#训练模型
loss_train_relu = []
loss_test_relu = []
acc_train_relu = []
acc_test_relu = []
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        X = X.view((-1, num_inputs)) 
        y_hat = net(X)
        l = loss(y_hat,y).sum()   
        optimizer.zero_grad()
        l.backward()       #计算损失     
        optimizer.step()   #更新梯度
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    loss_train_relu.append(train_l_sum/n)
    loss_test_relu.append(evaluate_loss(test_iter,net))
    acc_train_relu.append(train_acc_sum/n)
    acc_test_relu.append(evaluate_accuracy(test_iter,net))
    print('epoch %d, loss_train %f, loss_test %f, acc_train %f, acc_test %f '% 
          (epoch+1, loss_train_relu[epoch], loss_test_relu[epoch], acc_train_relu[epoch], acc_test_relu[epoch]))


# In[4]:


# ========== LeakyReLU激活函数 ==========

class LeakyReLU_Classification(torch.nn.Module):
    def __init__(self):
        super(LeakyReLU_Classification, self).__init__()
        self.hidden = torch.nn.Linear(num_inputs, num_hiddens) #定义隐藏层函数
        self.leakyrelu = torch.nn.LeakyReLU()                     #定义隐藏层激活函数
        self.output = torch.nn.Linear(num_hiddens, num_outputs)#定义输出层函数

    def forward(self, x):
        x = self.hidden(x)
        x = self.leakyrelu(x)
        x = self.output(x)
        return x
        
net = LeakyReLU_Classification()

#初始化模型参数
init.normal_(net.hidden.weight, mean=0, std=0.01)
init.normal_(net.output.weight, mean=0, std=0.01)
init.constant_(net.hidden.bias, val=0)
init.constant_(net.output.bias, val=0)

#定义损失函数和优化器
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr)

#训练模型
loss_train_leakyrelu = []
loss_test_leakyrelu = []
acc_train_leakyrelu = []
acc_test_leakyrelu = []
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        X = X.view((-1, num_inputs)) 
        y_hat = net(X)
        l = loss(y_hat,y).sum()  
        optimizer.zero_grad()
        l.backward()       #计算损失     
        optimizer.step()   #更新梯度
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    loss_train_leakyrelu.append(train_l_sum/n)
    loss_test_leakyrelu.append(evaluate_loss(test_iter,net))
    acc_train_leakyrelu.append(train_acc_sum/n)
    acc_test_leakyrelu.append(evaluate_accuracy(test_iter,net))
    print('epoch %d, loss_train %f, loss_test %f, acc_train %f, acc_test %f '% 
          (epoch+1, loss_train_leakyrelu[epoch], loss_test_leakyrelu[epoch], acc_train_leakyrelu[epoch], acc_test_leakyrelu[epoch]))


# In[5]:


# ========== Tanh激活函数 ==========

class Tanh_Classification(torch.nn.Module):
    def __init__(self):
        super(Tanh_Classification, self).__init__()
        self.hidden = torch.nn.Linear(num_inputs, num_hiddens) #定义隐藏层函数
        self.tanh = torch.nn.Tanh()                            #定义隐藏层激活函数
        self.output = torch.nn.Linear(num_hiddens, num_outputs)#定义输出层函数

    def forward(self, x):
        x = self.hidden(x)
        x = self.tanh(x)
        x = self.output(x)
        return x
        
net = Tanh_Classification()

#初始化模型参数
init.normal_(net.hidden.weight, mean=0, std=0.01)
init.normal_(net.output.weight, mean=0, std=0.01)
init.constant_(net.hidden.bias, val=0)
init.constant_(net.output.bias, val=0)

#定义损失函数和优化器
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr)

#训练模型
loss_train_tanh = []
loss_test_tanh = []
acc_train_tanh = []
acc_test_tanh = []
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        X = X.view((-1, num_inputs)) 
        y_hat = net(X)
        l = loss(y_hat,y).sum()  
        optimizer.zero_grad()
        l.backward()       #计算损失     
        optimizer.step()   #更新梯度
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    loss_train_tanh.append(train_l_sum/n)
    loss_test_tanh.append(evaluate_loss(test_iter,net))
    acc_train_tanh.append(train_acc_sum/n)
    acc_test_tanh.append(evaluate_accuracy(test_iter,net))
    print('epoch %d, loss_train %f, loss_test %f, acc_train %f, acc_test %f '% 
          (epoch+1, loss_train_tanh[epoch], loss_test_tanh[epoch], acc_train_tanh[epoch], acc_test_tanh[epoch]))


# In[15]:


plt.figure(figsize=(10,7))
plt.plot(loss_train_relu,label='relu',color='blue')
plt.plot(loss_train_leakyrelu,label='leakyrelu', color='orange', linestyle='--')
plt.plot(loss_train_tanh,label='tanh', color='red')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Train Loss")
plt.legend()
plt.show()


# In[16]:


plt.figure(figsize=(10,7))
plt.plot(loss_test_relu,label='relu',color='blue')
plt.plot(loss_test_leakyrelu,label='leakyrelu', color='orange', linestyle='--')
plt.plot(loss_test_tanh,label='tanh', color='red')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Test Loss")
plt.legend()
plt.show()


# In[19]:


plt.figure(figsize=(10,7))
plt.plot(acc_train_relu,label='relu',color='blue')
plt.plot(acc_train_leakyrelu,label='leakyrelu', color='orange')
plt.plot(acc_train_tanh,label='tanh', color='red', linestyle='--')
plt.ylabel("acc")
plt.xlabel("epoch")
plt.title("Train Acc")
plt.legend()
plt.show()


# In[20]:


plt.figure(figsize=(10,7))
plt.plot(acc_test_relu,label='relu',color='blue')
plt.plot(acc_test_leakyrelu,label='leakyrelu', color='orange')
plt.plot(acc_test_tanh,label='tanh', color='red', linestyle='--')
plt.ylabel("acc")
plt.xlabel("epoch")
plt.title("Test Acc")
plt.legend()
plt.show()


# In[ ]:




