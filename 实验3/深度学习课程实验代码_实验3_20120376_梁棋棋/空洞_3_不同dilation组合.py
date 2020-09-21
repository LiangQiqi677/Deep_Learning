#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import datetime

device = torch.device("cuda:0")

path = "./车辆分类数据集/bus/bus001.jpg"
train_features = torch.Tensor(np.array(Image.open(path).resize((100,100),Image.ANTIALIAS))/255).view(1, 100, 100, 3)
test_features = torch.Tensor(np.array(Image.open(path).resize((100,100),Image.ANTIALIAS))/255).view(1, 100, 100, 3)

# 处理客车数据，总共218张，前153张为训练集，后65张为测试集
pic_num = 0
for filename in os.listdir("./车辆分类数据集/bus/"): 
    pic_num = pic_num + 1
    path = "./车辆分类数据集/bus/"+ filename
    img = torch.Tensor(np.array(Image.open(path).resize((100,100),Image.ANTIALIAS))/255).view(1, 100, 100, 3)
    if pic_num <= 153:
        train_features = torch.cat((train_features,img), dim=0)
    else:
        test_features = torch.cat((test_features,img), dim=0)

# 处理汽车数据，总共779张，前545张为训练集，后234张为测试集
pic_num = 0
for filename in os.listdir("./车辆分类数据集/car/"): 
    pic_num = pic_num + 1
    path = "./车辆分类数据集/car/"+ filename
    img = torch.Tensor(np.array(Image.open(path).resize((100,100),Image.ANTIALIAS))/255).view(1, 100, 100, 3)
    if pic_num <= 545:
        train_features = torch.cat((train_features,img), dim=0)
    else:
        test_features = torch.cat((test_features,img), dim=0)

# 处理货车数据，总共360张，前252张为训练集，后108张为测试集
pic_num = 0
for filename in os.listdir("./车辆分类数据集/truck/"): 
    pic_num = pic_num + 1
    path = "./车辆分类数据集/truck/"+ filename
    img = torch.Tensor(np.array(Image.open(path).resize((100,100),Image.ANTIALIAS))/255).view(1, 100, 100, 3)
    if pic_num <= 252:
        train_features = torch.cat((train_features,img), dim=0)
    else:
        test_features = torch.cat((test_features,img), dim=0)

train_features = train_features.permute(0,3,2,1)
test_features = test_features.permute(0,3,2,1)
print(train_features.shape)
print(test_features.shape)

# =================== 训练集标签 =================== #
train_labels = torch.zeros(154).long()
train_labels = torch.cat((train_labels,torch.ones(545).long()), dim=0)
train_labels = torch.cat((train_labels,torch.ones(252).long()+1), dim=0)
print(train_labels.shape)

# =================== 测试集标签 =================== #
test_labels = torch.zeros(66).long()
test_labels = torch.cat((test_labels,torch.ones(234).long()), dim=0)
test_labels = torch.cat((test_labels,torch.ones(108).long()+1), dim=0)
print(test_labels.shape)


# In[7]:


# ==================== [1, 2, 3] ==================== #
class DilatedConvModule_1 (torch.nn.Module):
    def __init__(self):
        super(DilatedConvModule_1, self).__init__()
        #定义一个三层卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True) 
        )
        self.fc = nn.Linear(128, num_classes) #[32,3] ,32是卷积输出的通道数，out_channels是多少这里就是多少
 
    def forward(self, X):
        out = self.conv(X)
        #print(out.shape)  #torch.Size([32, 32, 84, 84])
        out = nn.functional.avg_pool2d(out, 84)
        #print(out.shape) #torch.Size([32, 32, 1, 1])
        out = out.squeeze()
        #print(out.shape) #torch.Size([32, 32])
        out = self.fc(out)
        return out

# ==================== [1, 2, 5] ==================== #
class DilatedConvModule_2 (torch.nn.Module):
    def __init__(self):
        super(DilatedConvModule_2, self).__init__()
        #定义一个三层卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True) 
        )
        self.fc = nn.Linear(128, num_classes) #[32,3] ,32是卷积输出的通道数，out_channels是多少这里就是多少
 
    def forward(self, X):
        out = self.conv(X)
        #print(out.shape)  #torch.Size([32, 32, 84, 84])
        out = nn.functional.avg_pool2d(out, 84)
        #print(out.shape) #torch.Size([32, 32, 1, 1])
        out = out.squeeze()
        #print(out.shape) #torch.Size([32, 32])
        out = self.fc(out)
        return out
    
# ==================== [3, 4, 5] ==================== #
class DilatedConvModule_3 (torch.nn.Module):
    def __init__(self):
        super(DilatedConvModule_3, self).__init__()
        #定义一个三层卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True) 
        )
        self.fc = nn.Linear(128, num_classes) #[32,3] ,32是卷积输出的通道数，out_channels是多少这里就是多少
 
    def forward(self, X):
        out = self.conv(X)
        #print(out.shape)  #torch.Size([32, 32, 84, 84])
        out = nn.functional.avg_pool2d(out, 76)
        #print(out.shape) #torch.Size([32, 32, 1, 1])
        out = out.squeeze()
        #print(out.shape) #torch.Size([32, 32])
        out = self.fc(out)
        return out


# In[3]:


#训练函数
def train_epoch(net, data_loader, device):
    
    net.train()
    train_batch_num = len(data_loader)
    total_loss = 0
    correct = 0
    sample_num = 0
    
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device).float()
        target = target.to(device).long()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        prediction = torch.argmax(output, 1)
        correct += (prediction == target).sum().item()
        sample_num += len(prediction)
    
    loss = total_loss / train_batch_num
    acc = correct / sample_num
    return loss, acc

#测试函数
def test_epoch(net, data_loader, device):
    
    net.eval()
    test_batch_num = len(data_loader)
    total_loss = 0
    correct = 0
    sample_num = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device).float()
            target = target.to(device).long()
            output = net(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            prediction = torch.argmax(output, 1)
            correct += (prediction == target).sum().item()
            sample_num += len(prediction)
    
    loss = total_loss / test_batch_num
    acc = correct / sample_num
    return loss, acc

num_classes = 3
num_epoch = 20
lr = 0.001
batch_size = 128

#划分数据集
train_dataset = torch.utils.data.TensorDataset(train_features,train_labels)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
test_dataset = torch.utils.data.TensorDataset(test_features,test_labels)
test_iter = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=True, num_workers=0)


# In[4]:


# ==================== [1, 2, 3] ==================== #

net = DilatedConvModule_1().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

train_loss_1 = []
train_acc_1 = []
test_loss_1 = []
test_acc_1 = []

start = datetime.datetime.now()
for epoch in range(num_epoch):
    
    train_loss, train_acc = train_epoch(net, train_iter, device)
    test_loss, test_acc = test_epoch(net, test_iter, device)
    
    train_loss_1.append(train_loss)
    train_acc_1.append(train_acc)
    test_loss_1.append(test_loss)
    test_acc_1.append(test_acc)
    
    print('epoch %d, train_loss %f, test_loss %f, train_acc %f, test_acc %f' % 
          (epoch+1, train_loss, test_loss, train_acc, test_acc))
end = datetime.datetime.now()
print("空洞卷积[1, 2, 3]执行时间：")
print(end - start)


# In[5]:


# ==================== [1, 2, 5] ==================== #

net = DilatedConvModule_2().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

train_loss_2 = []
train_acc_2 = []
test_loss_2 = []
test_acc_2 = []

start1 = datetime.datetime.now()
for epoch in range(num_epoch):
    
    train_loss, train_acc = train_epoch(net, train_iter, device)
    test_loss, test_acc = test_epoch(net, test_iter, device)
    
    train_loss_2.append(train_loss)
    train_acc_2.append(train_acc)
    test_loss_2.append(test_loss)
    test_acc_2.append(test_acc)
    
    print('epoch %d, train_loss %f, test_loss %f, train_acc %f, test_acc %f' % 
          (epoch+1, train_loss, test_loss, train_acc, test_acc))
end1 = datetime.datetime.now()
print("空洞卷积[1, 2, 5]执行时间：")
print(end1 - start1)


# In[8]:


# ==================== [3, 4, 5] ==================== #

net = DilatedConvModule_3().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

train_loss_3 = []
train_acc_3 = []
test_loss_3 = []
test_acc_3 = []

start3 = datetime.datetime.now()
for epoch in range(num_epoch):
    
    train_loss, train_acc = train_epoch(net, train_iter, device)
    test_loss, test_acc = test_epoch(net, test_iter, device)
    
    train_loss_3.append(train_loss)
    train_acc_3.append(train_acc)
    test_loss_3.append(test_loss)
    test_acc_3.append(test_acc)
    
    print('epoch %d, train_loss %f, test_loss %f, train_acc %f, test_acc %f' % 
          (epoch+1, train_loss, test_loss, train_acc, test_acc))
end3 = datetime.datetime.now()
print("空洞卷积[3, 4, 5]执行时间：")
print(end3 - start3)


# In[9]:


plt.plot(train_loss_1 ,label='Dlation[1,2,3]',color='blue')
plt.plot(train_loss_2 ,label='Dlation[1,2,5]', color='orange')
plt.plot(train_loss_3 ,label='Dlation[3,4,5]', color='red')
plt.ylabel("train_loss")
plt.xlabel("epoch")
plt.title("train_loss")
plt.legend()
plt.show()


# In[10]:


plt.plot(test_loss_1 ,label='Dlation[1,2,3]',color='blue')
plt.plot(test_loss_2 ,label='Dlation[1,2,5]', color='orange')
plt.plot(test_loss_3 ,label='Dlation[3,4,5]', color='red')
plt.ylabel("test_loss")
plt.xlabel("epoch")
plt.title("test_loss")
plt.legend()
plt.show()


# In[11]:


plt.plot(train_acc_1 ,label='Dlation[1,2,3]',color='blue')
plt.plot(train_acc_2 ,label='Dlation[1,2,5]', color='orange')
plt.plot(train_acc_3 ,label='Dlation[3,4,5]', color='red')
plt.ylabel("train_acc")
plt.xlabel("epoch")
plt.title("train_acc")
plt.legend()
plt.show()


# In[12]:


plt.plot(test_acc_1 ,label='Dlation[1,2,3]',color='blue')
plt.plot(test_acc_2 ,label='Dlation[1,2,5]', color='orange')
plt.plot(test_acc_3 ,label='Dlation[3,4,5]', color='red')
plt.ylabel("test_acc")
plt.xlabel("epoch")
plt.title("test_acc")
plt.legend()
plt.show()


# In[ ]:




