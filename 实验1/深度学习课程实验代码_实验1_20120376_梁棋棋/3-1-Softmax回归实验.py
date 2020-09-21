#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
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


# ========== 模型参数初始化 ==========

num_inputs = 784 #模型的输⼊向量的⻓度是28×28=784
num_outputs = 10 #图像有10个类别，即输出10个类别的概率，取最大

W = torch.normal(0, 0.01, (num_inputs, num_outputs), dtype=torch.float,requires_grad=True)
b = torch.zeros(num_outputs, dtype=torch.float,requires_grad=True)


# In[5]:


# ========== 定义softmax回归模型函数 ==========

def net(x):
    # [1,784]x[784,10]+[1,10] = [1,10]
    x_linear = torch.mm(x.view((-1, num_inputs)), W) + b  #线性模型
    x_exp = x_linear.exp()                                #softmax操作1
    x_sum = x_exp.sum(dim=1, keepdim=True)                #softmax操作2
    x_softmax = x_exp/x_sum                               #softmax操作3
    return x_softmax


# In[6]:


# ========== 定义交叉熵损失函数 ==========

def CrossEntropy_Loss(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))

#y_hat：每类标签的预测值
#y：标签的正确分类，假如分为第一类就是（1，0，0），分为第二类就是（0，1，0） 
#return一个标量，也就是loss
#假设有一个3分类问题，某个样例的正确分类是(1，0，0)(也就是说它是被分到第一类),假设经过softmax回归之后的预测概率是(0.5，0.4，0.1)
#那么预测和正确答案之间的交叉熵为：H((1，0，0),(0.5，0.4，0.1)) = -(1xlog0.5+0xlog0.4+0xlog0.1) = -(1xlog0.5) = 0.3
#由于分类只有一个1，其余都是0，所以就相当于只要计算分类为1的那类的概率即可（其余都是0）


# In[7]:


# ========== 定义net在数据集data_iter的准确率 ==========

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# In[8]:


# ========== 训练模型 ==========

#设置学习率和迭代次数
lr = 0.1
num_epochs = 100

loss_epoch = []
train_acc_epoch = []
test_acc_epoch = []

for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        y_hat = net(X)
        loss = CrossEntropy_Loss(y_hat,y).sum()
        loss.backward()
        W.data = W.data - lr * W.grad.data / batch_size #更新W
        W.grad.data.zero_()
        b.data = b.data - lr * b.grad.data / batch_size #更新b
        b.grad.data.zero_()

        train_l_sum += loss.item()
        train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
        n += y.shape[0]
        
    test_acc = evaluate_accuracy(test_iter, net)
    loss_epoch.append(train_l_sum/n)
    train_acc_epoch.append(train_acc_sum/n)
    test_acc_epoch.append(test_acc)
    print('epoch %d, loss %f, train_acc %f，test_acc %f'% (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc))        


# In[11]:


# ========== 损失函数可视化 ==========

plt.plot(loss_epoch)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title('Loss')
plt.show()


# In[12]:


# ========== 训练集准确率可视化 ==========

plt.plot(train_acc_epoch)
plt.ylabel("train_acc")
plt.xlabel("epoch")
plt.title('Train Accuracy Rate')
plt.show()


# In[13]:


# ========== 测试集准确率可视化 ==========

plt.plot(test_acc_epoch)
plt.ylabel("test_acc")
plt.xlabel("epoch")
plt.title('Test Accuracy Rate')
plt.show()


# In[ ]:




