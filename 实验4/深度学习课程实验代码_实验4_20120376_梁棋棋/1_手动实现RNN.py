#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import datetime
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# In[2]:


# ==================== 读取数据 ==================== #

df = pd.read_csv('FS_NYC.csv')                          #自动把第一行做列属性
df = df.loc[:,['userId','venueCategory']]               #取出用户ID和地点ID

# ==================== 获得地址字典 ==================== #

location = []                                           #取出所有的地址
for i in range(df.shape[0]):
    location.append(df.iloc[i,1])
location = list(set(location))                          #去重
loc_dict = dict(zip(location,range(len(location))))     #给这251个地点创建字典
#print(len(loc_dict))

# ==================== 分组获得每个人的运动轨迹 ==================== #

loc_rep = df.replace(loc_dict)                          #将地址名称替换成字典里的数字

def sliding_window(seq, window_size):
    result = []
    for i in range(len(seq)-window_size):
        result.append(seq[i:i+window_size])
    return result

train_set, test_set = [], []
window_size = 10                                        #窗口大小为10
for userId, group in loc_rep.groupby('userId'):
    full_seq = group['venueCategory']
    full_len = full_seq.shape[0]
    train_seq = full_seq.iloc[:int(full_len*0.7)].to_list()
    test_seq = full_seq.iloc[int(full_len*0.7):].to_list()
    train_set += sliding_window(train_seq, window_size)
    test_set += sliding_window(test_seq, window_size)
    
# loc_ind = loc_rep.set_index('venueCategory')            #将地址那一列设置成index，便于后续groups操作
# data_dict = loc_ind.groupby('userId').groups            #按照userId分组，将各个分组的index作为字典值
# dataset = []
# for key,value in data_dict.items():
#     dataset.append(list(value))


# In[3]:


# ==================== RNN模型 ==================== #

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        
        self.w_h = nn.Parameter(torch.rand(hidden_size, hidden_size)) #input_size是embedding_dim
        self.u_h = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        
        self.w_y = nn.Parameter(torch.rand(hidden_size, output_size))
        self.b_y = nn.Parameter(torch.zeros(output_size))
        
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = self.embedding(x) #(B, 10, 1024)
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        y_list = []
        for i in range(seq_len):
            h = self.tanh(torch.matmul(x[:, i, :], self.w_h) + torch.matmul(h, self.u_h) + self.b_h)
            y = self.leaky_relu(torch.matmul(h, self.w_y) + self.b_y)
            y_list.append(y)
        return h, torch.stack(y_list, dim=1)

def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]
    
def train_evaluate(model, data_set, batch_size, optimizer):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for batch in next_batch(shuffle(data_set), batch_size):
        batch = torch.from_numpy(np.array(batch)).long().to(device) #embedding层的输入需要long形式
        x = batch[:, :window_size-1].long().to(device)
        label =  batch[: , -1].long().to(device)
        hidden, out = model(x)                                 #输入直接是(batch_size, seq)，不用扩展最后一维
        prediction = out[:, -1, :].squeeze(-1)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_l_sum += loss.item()
        train_acc_sum += (prediction.argmax(dim=1) == label).sum().item() / label.shape[0]
        n += 1  #每个batch+1
    return train_l_sum/n, train_acc_sum/n

def test_evaluate(model, data_set, batch_size):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for batch in next_batch(shuffle(data_set), batch_size):
        batch = torch.from_numpy(np.array(batch)).long().to(device) #embedding层的输入需要long形式
        x = batch[:, :window_size-1].long().to(device)
        label =  batch[: , -1].long().to(device)
        hidden, out = model(x)                                 #输入直接是(batch_size, seq)，不用扩展最后一维
        prediction = out[:, -1, :].squeeze(-1)
        loss = loss_func(prediction, label)
        train_l_sum += loss.item()
        train_acc_sum += (prediction.argmax(dim=1) == label).sum().item() / label.shape[0]
        n += 1  #每个batch+1
    return train_l_sum/n, train_acc_sum/n


# In[5]:


device = 'cuda:1'
num_class = len(loc_dict)
model = MyRNN(input_size=num_class, hidden_size=1024, output_size=num_class).to(device) #输出size是地址类别

loss_func = nn.CrossEntropyLoss() #多分类问题使用CrossEntropyLoss函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,70], gamma=0.1)

loss_train = []
loss_test = []
acc_train = []
acc_test = []
batch_size = 4096
num_epoch = 70
start_total = datetime.datetime.now()
for epoch in range(num_epoch):
    start = datetime.datetime.now()
    train_l, train_acc = train_evaluate(model, train_set, batch_size, optimizer)
    test_l, test_acc = test_evaluate(model, test_set, batch_size)
    scheduler.step()
    loss_train.append(train_l)
    acc_train.append(train_acc)
    loss_test.append(test_l)
    acc_test.append(test_acc)
    end = datetime.datetime.now()
    print('epoch %d, loss_train %f, loss_test %f, acc_train %f, acc_test %f, time %s, lr %f'% 
          (epoch+1, loss_train[epoch], loss_test[epoch], acc_train[epoch], acc_test[epoch], end - start, 
           optimizer.param_groups[0]['lr']))
end_total = datetime.datetime.now()
print("手动实现RNN的运行时间：",end_total - start_total)


# In[6]:


plt.plot(loss_train ,label='train',color='royalblue')
plt.plot(loss_test ,label='test', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("RNN: Loss")
plt.legend()
plt.show()


# In[7]:


plt.plot(acc_train ,label='train',color='royalblue')
plt.plot(acc_test ,label='test', color='orange')
plt.ylabel("acc")
plt.xlabel("epoch")
plt.title("RNN: Acc")
plt.legend()
plt.show()


# In[5]:


data=[{"Time":"0:05:54.587870"},{"Time":"0:21:54.854169"}]
df = pd.DataFrame(data,columns=['Time'],index = ['手动实现RNN', '利用torch.nn实现RNN'])
df


# In[ ]:




