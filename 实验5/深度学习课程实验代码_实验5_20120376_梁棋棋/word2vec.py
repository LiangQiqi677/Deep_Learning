#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim
import jieba
import json
import datetime
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader, TensorDataset


# In[2]:


sample_list=[]
with open('virus_train.txt','r', encoding = 'utf-8') as file_open:
    data = json.load(file_open)
    stop_word = ['【', '】', ')', '(', '、', '，', '“', '”', '。', '\n', '《', '》', ' ', '-', '！', '？', '.', 
                 '\'', '[', ']','：', '/', '.', '"', ':', '’', '．', ',', '…', '?','；','（','）','@','_']
    n = 0
    for line in data:
        sample_content = line['content']                                         #取出内容
        for i in stop_word:                                                      #去掉标点等词
            sample_content = sample_content.replace(i, "")
        sentence_list=[word for word in jieba.cut(sample_content,cut_all=False)] #采用精确模式分词
        sample_list.append(sentence_list)                                        #将分好的词加入到sample_list中
        n += 1
        if n > 10:
            break
#print(sample_list)   #[['天使'], ['致敬', '心心', '小凡', '也', '要', '做好', '防护', '措施'...]...]

dictionary = gensim.corpora.Dictionary(sample_list)
sample_features = []                                                             #提取出corpus中每句话的词，不要次数
for sentence in sample_list:
    words = []
    for word in sentence:
        words.append(dictionary.token2id[word])
    sample_features.append(words)
voc_size = len(dictionary.token2id)                                              #获得字典大小
#print(sample_features)  #[[0], [15, 12, 10, 2, 16, 3, 18, 14, 8, 1, 5, 4, 15, 11, 9, 6, 17, 13, 7]...]

WINDOWS = 1                                                                      #取左右窗口的词作为context_word
pairs = []                                                                       #存放训练对
for sentence in sample_features:
    for center_word_index in range(len(sentence)):
        center_word_ix = sentence[center_word_index]
        for win in range(-WINDOWS, WINDOWS+1):
            contenx_word_index = center_word_index + win
            if 0 <= contenx_word_index <= len(sentence)-1 and contenx_word_index != center_word_index:
                context_word_ix = sentence[contenx_word_index]
                pairs.append((center_word_ix, context_word_ix))
#print('(中心词,背景词)对大小：', len(pairs))
#print('字典大小：', voc_size)
#print(pairs)


# In[3]:


class SkipGram(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden = torch.nn.Linear(self.vocab_size, self.embedding_size)
        self.predict = torch.nn.Linear(self.embedding_size, self.vocab_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.predict(x).view(1,-1)
        out = torch.nn.functional.softmax(x,dim=1)
        return out


# In[4]:


device = 'cuda:1'
model = SkipGram(voc_size, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fun = torch.nn.CrossEntropyLoss()
num_epoch = 20

losses = []
for epoch in range(num_epoch):
    batch_num, loss_sum = 0, 0.0
    for i, (center_ix, context_ix) in enumerate(pairs):
        data_x = torch.Tensor(np.eye(voc_size)[center_ix]).float().to(device)
        y_pred = model(data_x)
        y_true = torch.Tensor([context_ix]).long().to(device)
        loss = loss_fun(y_pred.view(1, -1), y_true)
        loss_sum += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_num += 1
    losses.append(loss_sum/batch_num)
    print('epoch %d, loss %.4f' % (epoch+1, losses[epoch]))

# model_filename='skipgram_{}.pkl'.format(date_str)
# torch.save(net.state_dict(),'saved_model/{}'.format(model_filename))
# print('model is saved as {}'.format(model_filename))


# In[ ]:




