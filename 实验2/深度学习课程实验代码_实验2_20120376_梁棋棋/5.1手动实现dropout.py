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


# In[4]:


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

#定义丢弃函数
def dropout(X, drop_prob):
    X = X.float()
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return torch.zeros_like(X)
    mask = (torch.rand(X.shape) < keep_prob).float()
    return mask * X / keep_prob

#定义linear层函数
def linear(X, w, b):
    return torch.matmul(X, w.t())+ b

#定义隐藏层激活函数
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

#定义分类模型
def net(X, drop_prob, is_training=True):
    X = X.view((-1, num_inputs))          
    linear_1 = linear(X, W1, b1)
    H1 = relu(linear_1)            #第一层隐藏层
    if is_training:
        H1 = dropout(H1, drop_prob)
    linear_2 = linear(H1, W2, b2)
    H2 = relu(linear_2)
    if is_training:
        H2 = dropout(H2, drop_prob)
    linear_3 = linear(H2, W3, b3)
    return linear_3

#定义随机梯度下降函数
def SGD(params, lr):
    for param in params:
        param.data -= lr * param.grad
        
#定义计算测试集损失函数
def evaluate_loss(data_iter, net):
    l, n = 0.0, 0
    for X, y in data_iter:
        l += loss(net(X, 0.0, is_training=False),y).sum().item()
        n += y.shape[0]
    return l / n

#定义计算测试集准确率函数
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        X = X.view((-1, num_inputs)) 
        acc_sum += (net(X, 0.0, is_training=False).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# In[5]:


# ==================== 无丢弃率 ==================== #

#定义惩罚权重
drop_prob0 = 0.0

#定义随机初始化模型参数的函数
W1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_inputs)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_hiddens)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, (num_outputs, num_hiddens)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
params = [W1, b1, W2, b2, W3, b3]

#训练模型
loss_train0 = []
loss_test0 = []
acc_train0 = []
acc_test0 = []
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X, drop_prob0, is_training=True)
        l = loss(y_hat,y).sum() 
        l.backward()          #计算损失     
        SGD(params, lr)       #更新梯度
        W1.grad.data.zero_()  #梯度清零
        b1.grad.data.zero_()
        W2.grad.data.zero_()
        b2.grad.data.zero_()
        W3.grad.data.zero_()
        b3.grad.data.zero_()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    loss_train0.append(train_l_sum/n)
    loss_test0.append(evaluate_loss(test_iter,net))
    acc_train0.append(train_acc_sum/n)
    acc_test0.append(evaluate_accuracy(test_iter,net))
    print('epoch %d, loss_train %f, loss_test %f , acc_train %f, acc_test %f '% 
          (epoch+1, loss_train0[epoch], loss_test0[epoch], acc_train0[epoch], acc_test0[epoch]))


# In[15]:


#plt.figure(figsize=(10,7))
plt.plot(loss_train0,label='Train',color='blue')
plt.plot(loss_test0,label='Test', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Loss:Overfitting")
plt.legend()
plt.show()


# In[16]:


#plt.figure(figsize=(10,7))
plt.plot(acc_train0,label='Train',color='blue')
plt.plot(acc_test0,label='Test', color='orange')
plt.ylabel("acc")
plt.xlabel("epoch")
plt.title("Acc:Overfitting")
plt.legend()
plt.show()


# In[10]:


# ========== 丢弃率为0.2 ==========

#定义丢弃率
drop_prob1 = 0.2

#定义随机初始化模型参数的函数
W1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_inputs)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_hiddens)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, (num_outputs, num_hiddens)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
params = [W1, b1, W2, b2, W3, b3]

#训练模型
loss_train1 = []
loss_test1 = []
acc_train1 = []
acc_test1 = []
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X, drop_prob1, is_training=True)
        l = loss(y_hat,y).sum() 
        l.backward()          #计算损失     
        SGD(params, lr)       #更新梯度
        W1.grad.data.zero_()  #梯度清零
        b1.grad.data.zero_()
        W2.grad.data.zero_()
        b2.grad.data.zero_()
        W3.grad.data.zero_()
        b3.grad.data.zero_()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    loss_train1.append(train_l_sum/n)
    loss_test1.append(evaluate_loss(test_iter,net))
    acc_train1.append(train_acc_sum/n)
    acc_test1.append(evaluate_accuracy(test_iter,net))
    print('epoch %d, loss_train %f, loss_test %f , acc_train %f, acc_test %f '% 
          (epoch+1, loss_train1[epoch], loss_test1[epoch], acc_train1[epoch], acc_test1[epoch]))


# In[18]:


#plt.figure(figsize=(10,7))
plt.plot(loss_train1,label='Train',color='blue')
plt.plot(loss_test1,label='Test', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Loss:Dropout 0.2")
plt.legend()
plt.show()


# In[17]:


#plt.figure(figsize=(10,7))
plt.plot(acc_train1,label='Train',color='blue')
plt.plot(acc_test1,label='Test', color='orange')
plt.ylabel("acc")
plt.xlabel("epoch")
plt.title("Acc:Dropout 0.2")
plt.legend()
plt.show()


# In[11]:


# ========== 丢弃率为0.5 ==========

#定义丢弃率
drop_prob2 = 0.5

#定义随机初始化模型参数的函数
W1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_inputs)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_hiddens)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, (num_outputs, num_hiddens)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
params = [W1, b1, W2, b2, W3, b3]

#训练模型
loss_train2 = []
loss_test2 = []
acc_train2 = []
acc_test2 = []
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X, drop_prob2, is_training=True)
        l = loss(y_hat,y).sum() 
        l.backward()          #计算损失     
        SGD(params, lr)       #更新梯度
        W1.grad.data.zero_()  #梯度清零
        b1.grad.data.zero_()
        W2.grad.data.zero_()
        b2.grad.data.zero_()
        W3.grad.data.zero_()
        b3.grad.data.zero_()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    loss_train2.append(train_l_sum/n)
    loss_test2.append(evaluate_loss(test_iter,net))
    acc_train2.append(train_acc_sum/n)
    acc_test2.append(evaluate_accuracy(test_iter,net))
    print('epoch %d, loss_train %f, loss_test %f , acc_train %f, acc_test %f '% 
          (epoch+1, loss_train2[epoch], loss_test2[epoch], acc_train2[epoch], acc_test2[epoch]))


# In[19]:


#plt.figure(figsize=(10,7))
plt.plot(loss_train2,label='Train',color='blue')
plt.plot(loss_test2,label='Test', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Loss:Dropout 0.5")
plt.legend()
plt.show()


# In[20]:


#plt.figure(figsize=(10,7))
plt.plot(acc_train2,label='Train',color='blue')
plt.plot(acc_test2,label='Test', color='orange')
plt.ylabel("acc")
plt.xlabel("epoch")
plt.title("Acc:Dropout 0.5")
plt.legend()
plt.show()


# In[12]:


# ========== 丢弃率为0.7 ==========

#定义丢弃率
drop_prob3 = 0.7

#定义随机初始化模型参数的函数
W1 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_inputs)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_hiddens)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens, dtype=torch.float, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, (num_outputs, num_hiddens)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
params = [W1, b1, W2, b2, W3, b3]

#训练模型
loss_train3 = []
loss_test3 = []
acc_train3 = []
acc_test3 = []
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X, drop_prob3, is_training=True)
        l = loss(y_hat,y).sum() 
        l.backward()          #计算损失     
        SGD(params, lr)       #更新梯度
        W1.grad.data.zero_()  #梯度清零
        b1.grad.data.zero_()
        W2.grad.data.zero_()
        b2.grad.data.zero_()
        W3.grad.data.zero_()
        b3.grad.data.zero_()
        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
    loss_train3.append(train_l_sum/n)
    loss_test3.append(evaluate_loss(test_iter,net))
    acc_train3.append(train_acc_sum/n)
    acc_test3.append(evaluate_accuracy(test_iter,net))
    print('epoch %d, loss_train %f, loss_test %f , acc_train %f, acc_test %f '% 
          (epoch+1, loss_train3[epoch], loss_test3[epoch], acc_train3[epoch], acc_test3[epoch]))


# In[21]:


#plt.figure(figsize=(10,7))
plt.plot(loss_train3,label='Train',color='blue')
plt.plot(loss_test3,label='Test', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Loss:Dropout 0.7")
plt.legend()
plt.show()


# In[22]:


#plt.figure(figsize=(10,7))
plt.plot(acc_train3,label='Train',color='blue')
plt.plot(acc_test3,label='Test', color='orange')
plt.ylabel("acc")
plt.xlabel("epoch")
plt.title("Acc:Dropout 0.7")
plt.legend()
plt.show()


# In[23]:


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


# In[24]:


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


# In[25]:


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


# In[26]:


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




