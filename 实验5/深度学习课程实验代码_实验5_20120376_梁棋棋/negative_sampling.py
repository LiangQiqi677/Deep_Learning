#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import collections 
import numpy as np
import random
import math
import json
import jieba
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset


# In[2]:


# ====================================== 获得数据集 ====================================== #
sample_list=[]
with open('virus_train.txt','r', encoding = 'utf-8') as file_open:
    data = json.load(file_open)
    stop_word = ['【', '】', ')', '(', '、', '，', '“', '”', '。', '\n', '《', '》', ' ', '-', '！', '？', '.', 
                 '\'', '[', ']','：', '/', '.', '"', ':', '’', '．', ',', '…', '?','；','（','）','@','_']
    for line in data:
        sample_content = line['content']                                         #取出内容
        for i in stop_word:                                                      #去掉标点等词
            sample_content = sample_content.replace(i, "")
        sentence_list=[word for word in jieba.cut(sample_content,cut_all=False)] #采用精确模式分词
        sample_list.append(sentence_list)                                        #将分好的词加入到sample_list中
#print(sample_list)   #[['天使'], ['致敬', '心心', '小凡', '也', '要', '做好', '防护', '措施'...]...]


# ====================================== 统计词频 ====================================== #
counter = collections.Counter([token for sentence in sample_list for token in sentence])  #统计词频
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))                              #只保留至少出现5次的词
#print(counter) #counter表示词频： {'致敬': 24, '也': 11, '要': 14, '做好': 5, '防护': 5,...}


# ====================================== 将词映射到索引 ====================================== #
token = [tk for tk, _ in counter.items()]
idx_to_token = {idx: tk for idx, tk in enumerate(token)}
token_to_idx = {tk: idx for idx, tk in enumerate(token)}
dataset = [[token_to_idx[sample_list[i][j]] for j in range(len(sample_list[i])) if sample_list[i][j] in token_to_idx] 
           for i in range(len(sample_list))]
num_tokens = sum([len(st) for st in dataset])
#print(dataset)       # [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11...]]
#print(idx_to_token)  # {0: '致敬', 1: '也', 2: '要', 3: '做好'....}
#print(token_to_idx)  # {'致敬': 0, '也': 1, '要': 2, '做好': 3....}
#print(num_tokens)    # 总共159个词（含重复），不含重复的是114个词


# In[3]:


def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:               # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

all_centers, all_contexts = get_centers_and_contexts(dataset, 1) #all_centers：中心词；all_contexts：背景词
# for center, context in zip(*get_centers_and_contexts(dataset, 1)):
#     print('center', center, 'has contexts', context)
#    center 1 has contexts [2]       center 2 has contexts [1, 3]    center 3 has contexts [2, 4]

def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights))) #表示总共有多少个词
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成k个词的索引作为噪声词。
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

sampling_weights = [counter[w]**0.75 for w in token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5) #K=5
#print(len(all_contexts))   #有157组背景词（总共159个词，但是由于有两个词就是一句话，没有周围词，所以不计入在内）
#print(len(all_negatives))  #有157组negatives词


# In[6]:


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return (self.centers[index], self.contexts[index], self.negatives[index])

    def __len__(self):
        return len(self.centers)
    
def batchify(data):
    """用作DataLoader的参数collate_fn: 输入是个长为batchsize的list, 
    list中的每个元素都是Dataset类调用__getitem__得到的结果
    """
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))

batch_size = 512
dataset = MyDataset(all_centers, all_contexts, all_negatives)
data_iter = DataLoader(dataset, batch_size, shuffle=True, collate_fn=batchify)

# ====================================== 定义模型 ====================================== #
class SkipGram(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embed_v = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embed_u = nn.Embedding(self.vocab_size, self.embedding_size)

    def forward(self, center, contexts_and_negatives):
        v = self.embed_v(center)
        u = self.embed_u(contexts_and_negatives)
        pred = torch.bmm(v, u.permute(0, 2, 1))
        return pred

# ====================================== 定义损失函数 ====================================== #
class SigmoidBinaryCrossEntropyLoss(nn.Module):
    def __init__(self): # none mean sum
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()
    def forward(self, inputs, targets, mask=None):
        """
        input – Tensor shape: (batch_size, len)
        target – Tensor of the same shape as input
        """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        res = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return res.mean(dim=1)


# In[7]:


device = 'cuda:1'
embed_size = 128
net = SkipGram(len(idx_to_token), embed_size).to(device)
loss = SigmoidBinaryCrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
num_epochs = 50
for epoch in range(num_epochs):
    start, l_sum, n = time.time(), 0.0, 0
    for batch in data_iter:
        center, context_negative, mask, label = [d.to(device) for d in batch]
        pred = net(center, context_negative)
        # 使用掩码变量mask来避免填充项对损失函数计算的影响
        l = (loss(pred.view(label.shape), label, mask) *
             mask.shape[1] / mask.float().sum(dim=1)).mean() # 一个batch的平均loss
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.cpu().item()
        n += 1
    print('epoch %d, loss %f, time %.6fs'% (epoch + 1, l_sum / n, time.time() - start))


# In[11]:


embedding_weights = net.embed_v.weight.data.cpu().numpy()
np.save("embedding-{}".format(embed_size), embedding_weights)
torch.save(net.state_dict(), "embedding-{}.th".format(embed_size))


# In[13]:


net.load_state_dict(torch.load("embedding-{}.th".format(embed_size)))


# In[ ]:




