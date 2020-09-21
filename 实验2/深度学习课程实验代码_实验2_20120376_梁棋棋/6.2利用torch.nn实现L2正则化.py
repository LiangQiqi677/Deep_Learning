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

#定义训练参数
batch_size, num_epochs, lr = 128, 20, 0.1

#划分数据集
temp = []
for i in range(5000):
    temp.append(list(mnist_train[i]))
train_iter = torch.utils.data.DataLoader(temp, batch_size=batch_size, shuffle=True,num_workers=0)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,num_workers=0)

#定义模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

#定义损失函数
loss = torch.nn.CrossEntropyLoss()

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


# In[3]:


# ========================== 惩罚权重为0，无L2正则化 ========================== #

lambd0 = 0

class Classification(torch.nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.hidden1 = torch.nn.Linear(num_inputs, num_hiddens) #定义隐藏层函数
        self.relu1 = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.hidden2 = torch.nn.Linear(num_hiddens, num_hiddens) #定义隐藏层函数
        self.relu2 = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.output = torch.nn.Linear(num_hiddens, num_outputs)#定义输出层函数

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x
        
net = Classification()

#初始化模型参数
init.normal_(net.hidden1.weight, mean=0, std=0.01)
init.normal_(net.hidden2.weight, mean=0, std=0.01)
init.normal_(net.output.weight, mean=0, std=0.01)
init.constant_(net.hidden1.bias, val=0)
init.constant_(net.hidden2.bias, val=0)
init.constant_(net.output.bias, val=0)

#定义优化器
optimizer_w1 = optim.SGD([net.hidden1.weight], lr, weight_decay=lambd0)
optimizer_w2 = optim.SGD([net.hidden2.weight], lr, weight_decay=lambd0)
optimizer_w3 = optim.SGD([net.output.weight], lr, weight_decay=lambd0)
optimizer_b1 = optim.SGD([net.hidden1.bias], lr)
optimizer_b2 = optim.SGD([net.hidden2.bias], lr)
optimizer_b3 = optim.SGD([net.output.bias], lr)

#训练模型
loss_train0 = []
loss_test0 = []
acc_train0 = []
acc_test0 = []
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        X = X.view((-1, num_inputs)) 
        y_hat = net(X)
        l = loss(y_hat,y).sum() 
        optimizer_w1.zero_grad()
        optimizer_w2.zero_grad()
        optimizer_w3.zero_grad()
        optimizer_b1.zero_grad()
        optimizer_b1.zero_grad()
        optimizer_b1.zero_grad()
        l.backward()       #计算损失     
        optimizer_w1.step()   #更新梯度
        optimizer_w2.step()
        optimizer_w3.step()
        optimizer_b1.step()   #更新梯度
        optimizer_b2.step()
        optimizer_b3.step()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    loss_train0.append(train_l_sum/n)
    loss_test0.append(evaluate_loss(test_iter,net))
    acc_train0.append(train_acc_sum/n)
    acc_test0.append(evaluate_accuracy(test_iter,net))
    print('epoch %d, loss_train %f, loss_test %f , acc_train %f, acc_test %f '% 
          (epoch+1, loss_train0[epoch], loss_test0[epoch], acc_train0[epoch], acc_test0[epoch]))


# In[4]:


print('L2 norm of W1: %f, L2 norm of W2: %f, L2 norm of W3: %f' % (
    net.hidden1.weight.norm().item(), net.hidden2.weight.norm().item(), net.output.weight.norm().item()))


# In[5]:


# ========================== 惩罚权重为1 ========================== #

lambd1 = 1

class Classification(torch.nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.hidden1 = torch.nn.Linear(num_inputs, num_hiddens) #定义隐藏层函数
        self.relu1 = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.hidden2 = torch.nn.Linear(num_hiddens, num_hiddens) #定义隐藏层函数
        self.relu2 = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.output = torch.nn.Linear(num_hiddens, num_outputs)#定义输出层函数

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x
        
net = Classification()

#初始化模型参数
init.normal_(net.hidden1.weight, mean=0, std=0.01)
init.normal_(net.hidden2.weight, mean=0, std=0.01)
init.normal_(net.output.weight, mean=0, std=0.01)
init.constant_(net.hidden1.bias, val=0)
init.constant_(net.hidden2.bias, val=0)
init.constant_(net.output.bias, val=0)

#定义优化器
optimizer_w1 = optim.SGD([net.hidden1.weight], lr, weight_decay=lambd1)
optimizer_w2 = optim.SGD([net.hidden2.weight], lr, weight_decay=lambd1)
optimizer_w3 = optim.SGD([net.output.weight], lr, weight_decay=lambd1)
optimizer_b1 = optim.SGD([net.hidden1.bias], lr)
optimizer_b2 = optim.SGD([net.hidden2.bias], lr)
optimizer_b3 = optim.SGD([net.output.bias], lr)

#训练模型
loss_train1 = []
loss_test1 = []
acc_train1 = []
acc_test1 = []
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        X = X.view((-1, num_inputs)) 
        y_hat = net(X)
        l = loss(y_hat,y).sum() 
        optimizer_w1.zero_grad()
        optimizer_w2.zero_grad()
        optimizer_w3.zero_grad()
        optimizer_b1.zero_grad()
        optimizer_b1.zero_grad()
        optimizer_b1.zero_grad()
        l.backward()       #计算损失     
        optimizer_w1.step()   #更新梯度
        optimizer_w2.step()
        optimizer_w3.step()
        optimizer_b1.step()   #更新梯度
        optimizer_b2.step()
        optimizer_b3.step()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    loss_train1.append(train_l_sum/n)
    loss_test1.append(evaluate_loss(test_iter,net))
    acc_train1.append(train_acc_sum/n)
    acc_test1.append(evaluate_accuracy(test_iter,net))
    print('epoch %d, loss_train %f, loss_test %f , acc_train %f, acc_test %f '% 
          (epoch+1, loss_train1[epoch], loss_test1[epoch], acc_train1[epoch], acc_test1[epoch]))
print('L2 norm of W1: %f, L2 norm of W2: %f, L2 norm of W3: %f' % (
    net.hidden1.weight.norm().item(), net.hidden2.weight.norm().item(), net.output.weight.norm().item()))


# In[6]:


plt.plot(loss_train0,label='Train',color='blue')
plt.plot(loss_test0,label='Test', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Loss:Lambda=0")
plt.legend()
plt.show()


# In[7]:


plt.plot(loss_train1,label='Train',color='blue')
plt.plot(loss_test1,label='Test', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Loss:Lambda=1")
plt.legend()
plt.show()


# In[8]:


# ========================== 惩罚权重为2 ========================== #

lambd2 = 2

class Classification(torch.nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.hidden1 = torch.nn.Linear(num_inputs, num_hiddens) #定义隐藏层函数
        self.relu1 = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.hidden2 = torch.nn.Linear(num_hiddens, num_hiddens) #定义隐藏层函数
        self.relu2 = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.output = torch.nn.Linear(num_hiddens, num_outputs)#定义输出层函数

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x
        
net = Classification()

#初始化模型参数
init.normal_(net.hidden1.weight, mean=0, std=0.01)
init.normal_(net.hidden2.weight, mean=0, std=0.01)
init.normal_(net.output.weight, mean=0, std=0.01)
init.constant_(net.hidden1.bias, val=0)
init.constant_(net.hidden2.bias, val=0)
init.constant_(net.output.bias, val=0)

#定义优化器
optimizer_w1 = optim.SGD([net.hidden1.weight], lr, weight_decay=lambd2)
optimizer_w2 = optim.SGD([net.hidden2.weight], lr, weight_decay=lambd2)
optimizer_w3 = optim.SGD([net.output.weight], lr, weight_decay=lambd2)
optimizer_b1 = optim.SGD([net.hidden1.bias], lr)
optimizer_b2 = optim.SGD([net.hidden2.bias], lr)
optimizer_b3 = optim.SGD([net.output.bias], lr)

#训练模型
loss_train2 = []
loss_test2 = []
acc_train2 = []
acc_test2 = []
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        X = X.view((-1, num_inputs)) 
        y_hat = net(X)
        l = loss(y_hat,y).sum() 
        optimizer_w1.zero_grad()
        optimizer_w2.zero_grad()
        optimizer_w3.zero_grad()
        optimizer_b1.zero_grad()
        optimizer_b1.zero_grad()
        optimizer_b1.zero_grad()
        l.backward()       #计算损失     
        optimizer_w1.step()   #更新梯度
        optimizer_w2.step()
        optimizer_w3.step()
        optimizer_b1.step()   #更新梯度
        optimizer_b2.step()
        optimizer_b3.step()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    loss_train2.append(train_l_sum/n)
    loss_test2.append(evaluate_loss(test_iter,net))
    acc_train2.append(train_acc_sum/n)
    acc_test2.append(evaluate_accuracy(test_iter,net))
    print('epoch %d, loss_train %f, loss_test %f , acc_train %f, acc_test %f '% 
          (epoch+1, loss_train2[epoch], loss_test2[epoch], acc_train2[epoch], acc_test2[epoch]))
print('L2 norm of W1: %f, L2 norm of W2: %f, L2 norm of W3: %f' % (
    net.hidden1.weight.norm().item(), net.hidden2.weight.norm().item(), net.output.weight.norm().item()))


# In[9]:


plt.plot(loss_train2,label='Train',color='blue')
plt.plot(loss_test2,label='Test', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Loss:Lambda=2")
plt.legend()
plt.show()


# In[ ]:




