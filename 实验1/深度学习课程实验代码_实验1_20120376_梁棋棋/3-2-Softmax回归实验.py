#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.nn import init
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# In[2]:


# ========== 加载数据集 ==========

mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True,
download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False,
download=True, transform=transforms.ToTensor())


# In[3]:


# ========== 读取小批量数据样本 ==========

batch_size = 256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,num_workers=0)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,num_workers=0)


# In[4]:


# ========== 构建模型和参数初始化 ==========

num_inputs = 784 #模型的输⼊向量的⻓度是28×28=784
num_outputs = 10 #图像有10个类别，即输出10个类别的概率，取最大

class LinearNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(num_inputs,num_outputs) #和Logister不一样，num_outputs不是1而是10
    
    # forward 定义前向传播
    def forward(self,x):  
        return self.linear(x.view(x.shape[0],-1))

net = LinearNet(num_inputs,num_outputs)

# ========== 模型参数初始化 ==========
nn.init.normal_(net.linear.weight,mean=0,std=0.01)
nn.init.constant_(net.linear.bias,val=0)


# In[5]:


# ========== 定义损失函数和优化器 ==========

loss = nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = optim.SGD(net.parameters(),lr=0.1) #优化器定义


# In[6]:


# ========== 定义net在数据集data_iter的准确率 ==========

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# In[7]:


# ========== 训练模型 ==========

#设置学习率和迭代次数
lr = 0.01
num_epochs = 100

loss_epoch = []
train_acc_epoch = []
test_acc_epoch = []

for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y).sum()
        l.backward()
        optimizer.step()         #更新
        optimizer.zero_grad()    #清空

        train_l_sum += l.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
        
    test_acc = evaluate_accuracy(test_iter, net)
    loss_epoch.append(train_l_sum/n)
    train_acc_epoch.append(train_acc_sum/n)
    test_acc_epoch.append(test_acc)
    print('epoch %d, loss %f, train_acc %f，test_acc %f'% (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc))        


# In[8]:


# ========== 损失函数可视化 ==========

plt.plot(loss_epoch)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title('Loss')
plt.show()


# In[9]:


# ========== 训练集准确率可视化 ==========

plt.plot(train_acc_epoch)
plt.ylabel("train_acc")
plt.xlabel("epoch")
plt.title('Train Accuracy Rate')
plt.show()


# In[10]:


# ========== 测试集准确率可视化 ==========

plt.plot(test_acc_epoch)
plt.ylabel("test_acc")
plt.xlabel("epoch")
plt.title('Test Accuracy Rate')
plt.show()


# In[ ]:




