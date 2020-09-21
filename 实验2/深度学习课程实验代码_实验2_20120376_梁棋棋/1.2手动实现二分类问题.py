#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt


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

#print(train_features.shape)
#print(test_features.shape)
#print(train_labels)
#print(test_labels)
#print(n_data.shape)
# print(x1.shape)
# print(y1.shape)
# print(x2.shape)
# print(y2.shape)


# In[3]:


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

#定义输出层激活函数
#Sigmoid函数用作二分类问题的意义：输出结果只有一个，如果值在0.5-1之间，表示分为正类；如果值在0-0.5之间，表示分为负例
#Softmax函数用作n分类问题的意义：输出结果有n个，这n个标签的概率加起来为1，哪个标签概率高就分为哪一类
#当n=2时，Sigmoid函数与Softmax函数相同
def sigmoid(X):
    return 1/(1+torch.exp(-X))  

#定义分类模型
def net(X):
    X = X.view((-1, num_inputs))          
    linear_1 = linear(X, W1, b1)
    R = relu(linear_1)
    linear_2 = linear(R, W2, b2)
    S = sigmoid(linear_2) 
    return S

#定义交叉熵损失函数（二分类任务可使用交叉熵损失函数）
def CrossEntropy_loss(y_hat,y):
    return -torch.mean(y.view(-1,1)*torch.log(y_hat) + (1-y.view(-1,1))*torch.log(1-y_hat))

#定义随机梯度下降函数
def SGD(params, lr):
    for param in params:
        param.data -= lr * param.grad


# In[4]:


#定义训练参数
batch_size, num_epochs, lr = 256, 100, 0.003

#划分数据集
dataset = torch.utils.data.TensorDataset(train_features,train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

#训练模型
loss_train = []
loss_test = []
for epoch in range(num_epochs):
    for X, y in train_iter:
        loss = CrossEntropy_loss(net(X),y).sum() 
        loss.backward()       #计算损失     
        SGD(params, lr)       #更新梯度
        W1.grad.data.zero_()
        b1.grad.data.zero_()
        W2.grad.data.zero_()
        b2.grad.data.zero_()
    loss_train.append(CrossEntropy_loss(net(train_features),train_labels).mean().item()) #训练集损失
    loss_test.append(CrossEntropy_loss(net(test_features),test_labels).mean().item())    #测试集损失
    print('epoch %d, loss_train %f, loss_test %f '% (epoch+1, loss_train[epoch], loss_test[epoch]))


# In[5]:


plt.figure(figsize=(10,7))
plt.plot(loss_train,label='train',color='royalblue')
plt.plot(loss_test,label='test',linestyle=':', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.show()


# In[ ]:




