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
batch_size, num_epochs, lr = 64, 100, 0.1

#划分数据集
temp = []
for i in range(1000):
    temp.append(list(mnist_train[i]))
train_iter = torch.utils.data.DataLoader(temp, batch_size=batch_size, shuffle=True,num_workers=0)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,num_workers=0)

#定义模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

#定义损失函数
loss = torch.nn.CrossEntropyLoss()

#定义计算测试集损失函数 isinstance(condi1,condi2),condi1:判断的数据；condi2:判断条件
def evaluate_loss(data_iter, net):
    l, n = 0.0, 0
    for X, y in data_iter:
        X = X.view((-1, num_inputs)) 
        if isinstance(net, torch.nn.Module):
            net.eval()
            l += loss(net(X),y).sum().item()
            net.train()
        n += y.shape[0]
    return l / n

#定义计算测试集准确率函数 isinstance(condi1,condi2),condi1:判断的数据；condi2:判断条件
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        X = X.view((-1, num_inputs)) 
        if isinstance(net, torch.nn.Module):
            net.eval()
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train()
        n += y.shape[0]
    return acc_sum / n


# In[3]:


# ========================== 无丢弃率，研究过拟合 ========================== #

dropout0 = 0.0

class Classification(torch.nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.hidden1 = torch.nn.Linear(num_inputs, num_hiddens) #定义隐藏层函数
        self.relu1 = torch.nn.ReLU()                       #定义隐藏层激活函数
        #self.dropout1 = torch.nn.Dropout()
        self.hidden2 = torch.nn.Linear(num_hiddens, num_hiddens) #定义隐藏层函数
        self.relu2 = torch.nn.ReLU()                       #定义隐藏层激活函数
        #self.dropout2 = torch.nn.Dropout()
        self.output = torch.nn.Linear(num_hiddens, num_outputs)#定义输出层函数

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        #x = self.dropout1(x,dropout1)
        x = self.hidden2(x)
        x = self.relu2(x)
        #x = self.dropout2(x,dropout1)
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
optimizer = optim.SGD(net.parameters(), lr)

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
        optimizer.zero_grad()
        l.backward()       #计算损失     
        optimizer.step()   #更新梯度
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


#plt.figure(figsize=(10,7))
plt.plot(loss_train0,label='Train',color='blue')
plt.plot(loss_test0,label='Test', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Loss:Overfitting")
plt.legend()
plt.show()


# In[5]:


#plt.figure(figsize=(10,7))
plt.plot(acc_train0,label='Train',color='blue')
plt.plot(acc_test0,label='Test', color='orange')
plt.ylabel("acc")
plt.xlabel("epoch")
plt.title("Acc:Overfitting")
plt.legend()
plt.show()


# In[9]:


# ========================== 丢弃率为0.2 ========================== #

dropout1 = 0.2

class Classification(torch.nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.hidden1 = torch.nn.Linear(num_inputs, num_hiddens) #定义隐藏层函数
        self.relu1 = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.dropout_1 = torch.nn.Dropout(dropout1)
        self.hidden2 = torch.nn.Linear(num_hiddens, num_hiddens) #定义隐藏层函数
        self.relu2 = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.dropout_2 = torch.nn.Dropout(dropout1)
        self.output = torch.nn.Linear(num_hiddens, num_outputs)#定义输出层函数

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.dropout_1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.dropout_2(x)
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
optimizer = optim.SGD(net.parameters(), lr)

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
        optimizer.zero_grad()
        l.backward()       #计算损失     
        optimizer.step()   #更新梯度
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    loss_train1.append(train_l_sum/n)
    loss_test1.append(evaluate_loss(test_iter,net))
    acc_train1.append(train_acc_sum/n)
    acc_test1.append(evaluate_accuracy(test_iter,net))
    print('epoch %d, loss_train %f, loss_test %f , acc_train %f, acc_test %f '% 
          (epoch+1, loss_train1[epoch], loss_test1[epoch], acc_train1[epoch], acc_test1[epoch]))


# In[10]:


#plt.figure(figsize=(10,7))
plt.plot(loss_train1,label='Train',color='blue')
plt.plot(loss_test1,label='Test', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Loss:Dropout 0.2")
plt.legend()
plt.show()


# In[11]:


#plt.figure(figsize=(10,7))
plt.plot(acc_train1,label='Train',color='blue')
plt.plot(acc_test1,label='Test', color='orange')
plt.ylabel("acc")
plt.xlabel("epoch")
plt.title("Acc:Dropout 0.2")
plt.legend()
plt.show()


# In[12]:


# ========================== 丢弃率为0.5 ========================== #

dropout2 = 0.5

class Classification(torch.nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.hidden1 = torch.nn.Linear(num_inputs, num_hiddens) #定义隐藏层函数
        self.relu1 = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.dropout_1 = torch.nn.Dropout(dropout2)
        self.hidden2 = torch.nn.Linear(num_hiddens, num_hiddens) #定义隐藏层函数
        self.relu2 = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.dropout_2 = torch.nn.Dropout(dropout2)
        self.output = torch.nn.Linear(num_hiddens, num_outputs)#定义输出层函数

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.dropout_1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.dropout_2(x)
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
optimizer = optim.SGD(net.parameters(), lr)

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
        optimizer.zero_grad()
        l.backward()       #计算损失     
        optimizer.step()   #更新梯度
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    loss_train2.append(train_l_sum/n)
    loss_test2.append(evaluate_loss(test_iter,net))
    acc_train2.append(train_acc_sum/n)
    acc_test2.append(evaluate_accuracy(test_iter,net))
    print('epoch %d, loss_train %f, loss_test %f , acc_train %f, acc_test %f '% 
          (epoch+1, loss_train2[epoch], loss_test2[epoch], acc_train2[epoch], acc_test2[epoch]))


# In[13]:


#plt.figure(figsize=(10,7))
plt.plot(loss_train2,label='Train',color='blue')
plt.plot(loss_test2,label='Test', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Loss:Dropout 0.5")
plt.legend()
plt.show()


# In[14]:


#plt.figure(figsize=(10,7))
plt.plot(acc_train2,label='Train',color='blue')
plt.plot(acc_test2,label='Test', color='orange')
plt.ylabel("acc")
plt.xlabel("epoch")
plt.title("Acc:Dropout 0.5")
plt.legend()
plt.show()


# In[15]:


# ========================== 丢弃率为0.7 ========================== #

dropout3 = 0.7

class Classification(torch.nn.Module):
    def __init__(self):
        super(Classification, self).__init__()
        self.hidden1 = torch.nn.Linear(num_inputs, num_hiddens) #定义隐藏层函数
        self.relu1 = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.dropout_1 = torch.nn.Dropout(dropout3)
        self.hidden2 = torch.nn.Linear(num_hiddens, num_hiddens) #定义隐藏层函数
        self.relu2 = torch.nn.ReLU()                       #定义隐藏层激活函数
        self.dropout_2 = torch.nn.Dropout(dropout3)
        self.output = torch.nn.Linear(num_hiddens, num_outputs)#定义输出层函数

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.dropout_1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.dropout_2(x)
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
optimizer = optim.SGD(net.parameters(), lr)

#训练模型
loss_train3 = []
loss_test3 = []
acc_train3 = []
acc_test3 = []
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
    loss_train3.append(train_l_sum/n)
    loss_test3.append(evaluate_loss(test_iter,net))
    acc_train3.append(train_acc_sum/n)
    acc_test3.append(evaluate_accuracy(test_iter,net))
    print('epoch %d, loss_train %f, loss_test %f , acc_train %f, acc_test %f '% 
          (epoch+1, loss_train3[epoch], loss_test3[epoch], acc_train3[epoch], acc_test3[epoch]))


# In[16]:


#plt.figure(figsize=(10,7))
plt.plot(loss_train3,label='Train',color='blue')
plt.plot(loss_test3,label='Test', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Loss:Dropout 0.7")
plt.legend()
plt.show()


# In[17]:


#plt.figure(figsize=(10,7))
plt.plot(acc_train3,label='Train',color='blue')
plt.plot(acc_test3,label='Test', color='orange')
plt.ylabel("acc")
plt.xlabel("epoch")
plt.title("Acc:Dropout 0.7")
plt.legend()
plt.show()


# In[18]:


#plt.figure(figsize=(10,7))
plt.plot(loss_train0,label='Overfitting',color='blue')
plt.plot(loss_train1,label='Dropout 0.2', color='orange')
plt.plot(loss_train2,label='Dropout 0.5', color='red')
plt.plot(loss_train3,label='Dropout 0.7', color='green')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Train Loss")
plt.legend()
plt.show()


# In[19]:


#plt.figure(figsize=(10,7))
plt.plot(loss_test0,label='Overfitting',color='blue')
plt.plot(loss_test1,label='Dropout 0.2', color='orange')
plt.plot(loss_test2,label='Dropout 0.5', color='red')
plt.plot(loss_test3,label='Dropout 0.7', color='green')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Test Loss")
plt.legend()
plt.show()


# In[20]:


#plt.figure(figsize=(10,7))
plt.plot(acc_train0,label='Overfitting',color='blue')
plt.plot(acc_train1,label='Dropout 0.2', color='orange')
plt.plot(acc_train2,label='Dropout 0.5', color='red')
plt.plot(acc_train3,label='Dropout 0.7', color='green')
plt.ylabel("acc")
plt.xlabel("epoch")
plt.title("Train Acc")
plt.legend()
plt.show()


# In[21]:


#plt.figure(figsize=(10,7))
plt.plot(acc_test0,label='Overfitting',color='blue')
plt.plot(acc_test1,label='Dropout 0.2', color='orange')
plt.plot(acc_test2,label='Dropout 0.5', color='red')
plt.plot(acc_test3,label='Dropout 0.7', color='green')
plt.ylabel("acc")
plt.xlabel("epoch")
plt.title("Test Acc")
plt.legend()
plt.show()


# In[ ]:




