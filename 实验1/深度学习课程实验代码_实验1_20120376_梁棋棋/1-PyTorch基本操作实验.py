#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torchvision

print(torch.__version__)


# ### 使用𝐓𝐞𝐧𝐬𝐨r初始化一个𝟏×𝟑的矩阵𝑴和一个𝟐×𝟏的矩阵𝑵，对两矩阵进行减法操作（要求实现三种不同的形式）

# In[3]:


M = torch.rand(1,3)
N = torch.rand(2,1)
print(M)
print(N)

#减法形式一
print(M-N)

#减法形式二
print(torch.sub(M,N))

#减法形式三
M.sub_(N)
print(M)


# ###  利用𝐓𝐞𝐧𝐬𝐨𝐫创建两个大小分别𝟑×𝟐和𝟒×𝟐的随机数矩阵𝑷和𝑸，要求服从均值为0，标准差0.01为的正态分布

# In[9]:


P = torch.normal(0,0.01,(3,2))
Q = torch.normal(0,0.01,(4,2))

print(P)
print(Q)

QT = torch.transpose(Q,0,1) #对Q进行转置
print(QT)

print(torch.mm(P, QT))


# ### 给定公式𝑦3=𝑦1+𝑦2=𝑥2+𝑥3，且𝑥=1。求𝑦3对𝑥的梯度

# In[44]:


x = torch.tensor(1.0,requires_grad=True) #要float的形式
print(x)

#计算𝑥^3时中断梯度的追踪
with torch.no_grad(): 
    y2 = x**3

y1 = x**2
y3 = y1+y2
y3.backward(x)
print(y3)
print(x.grad)

