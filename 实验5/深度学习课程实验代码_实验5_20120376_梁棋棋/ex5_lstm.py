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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


# In[2]:


#设置labels标签字典，用于替换labels
labels_dic = {'neural':0, 'happy':1, 'angry':2, 'sad':3, 'fear':4, 'surprise':5}

def get_dict(file_name):
    sample_list=[]
    sample_labels = []
    with open(file_name,'r', encoding = 'utf-8') as file_open:
        data = json.load(file_open)
        stop_word = ['【', '】', ')', '(', '、', '，', '“', '”', '。', '\n', '《', '》', ' ', '-', '！', '？', '.', 
                 '\'', '[', ']','：', '/', '.', '"', ':', '’', '．', ',', '…', '?','；','（','）','@','_']
        for line in data:
            sample_content = line['content']                                         #取出内容
            for i in stop_word:                                                      #去掉标点等词
                sample_content = sample_content.replace(i, "Stop")
            sentence_list=[word for word in jieba.cut(sample_content,cut_all=False)] #采用精确模式分词
            sample_list.append(sentence_list)                                        #将分好的词加入到sample_list中
            for i in range(len(sentence_list)):
                if 'Stop' in sentence_list[i]:
                    sentence_list[i] = 'Stop'
            sample_labels.append([labels_dic[line['label']]])                        #将labels用标签字典替换成数字
    #dictionary = gensim.corpora.Dictionary(sample_list)                              #dictionary：词典
    return sample_list, sample_labels

def get_dataset(dictionary, sample_list, sample_labels):
    sample_features = []                                                #提取出corpus中每句话的词，不要次数
    for sentence in sample_list:
        words = []
        for word in sentence:
            words.append(dictionary[word])
        sample_features.append(words)                                   

    sample_data = []                                                    #为了能够将train_data放到DataLoader中
    for i in range(len(sample_features)):
        temp = []
        temp.append(torch.Tensor(sample_features[i]).long())
        temp.append(torch.Tensor(sample_labels[i]).long())
        sample_data.append(temp)
    
    return sample_data


# dictionary, train_list, train_labels = get_dict('virus_train.txt')
# dictionary_test, test_list, test_labels = get_dict('virus_eval_labeled.txt')
# dict2_to_dict1 = dictionary.merge_with(dictionary_test)                              #将训练集和测试集的词典合并

train_list, train_labels = get_dict('virus_train.txt')
test_list, test_labels = get_dict('virus_eval_labeled.txt')

#model_word2vec = Word2Vec(sample_list, sg=1, size=128,  window=5,  min_count=3, sample=0.001)
#model_word2vec.save('word2vec_model.txt')
w2v_model = Word2Vec.load('word2vec_model.txt')
vocab_list = list(w2v_model.wv.vocab.keys())
word_index = {word: index for index, word in enumerate(vocab_list)}  #获得字典：{'天使': 0, '致敬': 1...}
train_data = get_dataset(word_index, train_list, train_labels)
test_data = get_dataset(word_index, test_list, test_labels)


# In[3]:


class Lstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embeding_vector):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embeding_vector))  #使用预训练的词向量
        self.embedding.weight.requires_grad = False
        #一定要加batch_first=True，因为我们的输入x是[batch,seq,input],但是默认的是[seq,batch,input]
        self.lstm = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True, dropout=0.1)
        self.out = torch.nn.Linear(hidden_size,output_size)
        
    def forward(self, x, batch_seq_len):
        batch_size = x.size(0)
        seq_len = x.size(1)                                           # x: torch.Size([128, 10])
        x = self.embedding(x)                                         # x: torch.Size([128, 10, 32])
        h = torch.zeros(1, batch_size, self.hidden_size).to(x.device) # h: #torch.Size([1, 128, 128])  
        c = torch.zeros(1, batch_size, self.hidden_size).to(x.device) # h: #torch.Size([1, 128, 128])
        
        x = rnn_utils.pack_padded_sequence(x,batch_seq_len, batch_first=True)
        output, hidden = self.lstm(x, (h, c))
        output, _ = rnn_utils.pad_packed_sequence(output,batch_first=True)
        output = self.out(output)
        #print(output.shape) #torch.Size([2, 23, 6])
        return hidden, output[:, -1, :]  #torch.Size([2, 6]) 直接输出最后一列

padding_value = word_index['Stop']
def collate_fn(sample_data):
    sample_data.sort(key=lambda data: len(data[0]), reverse=True)                          #倒序排序
    sample_features = []
    sample_labels = []
    for data in sample_data:
        sample_features.append(data[0])
        sample_labels.append(data[1])
    data_length = [len(data[0]) for data in sample_data]                                   #取出所有data的长度             
    sample_features = rnn_utils.pad_sequence(sample_features, batch_first=True, padding_value=padding_value) 
    return sample_features, sample_labels, data_length

def test_evaluate(model, test_dataloader, batch_size, num_epoch):
    test_l, test_a, test_p, test_r, test_f, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for data_x, data_y, batch_seq_len in test_dataloader:
            label = torch.Tensor(data_y).long().to(device)
            hidden, out = model(data_x.to(device),batch_seq_len)
            prediction = out.argmax(dim=1)
            loss = loss_func(out, label)
            prediction = out.argmax(dim=1).data.cpu().numpy()
            label = label.data.cpu().numpy()
            test_l += loss.item()
            test_a += accuracy_score(label, prediction)
            test_p += precision_score(label, prediction, average='weighted', labels=np.unique(prediction))
            test_r += recall_score(label, prediction, average='weighted', labels=np.unique(prediction))
            test_f += f1_score(label, prediction, average='weighted', labels=np.unique(prediction))
            n += 1
    return test_l/n, test_a/n, test_p/n, test_r/n, test_f/n


# In[26]:


device = 'cuda:2'
loss_func = torch.nn.CrossEntropyLoss()

# 让Embedding层使用训练好的Word2Vec权重
embedding_matrix = w2v_model.wv.vectors
input_size = embedding_matrix.shape[0]   #7748, 词典的大小
hidden_size = embedding_matrix.shape[1]  #128, 隐藏层单元个数
model = Lstm(input_size=input_size, hidden_size=hidden_size, output_size=6, embeding_vector=embedding_matrix).to(device)
model.load_state_dict(torch.load('./ex5_0.pt'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, weight_decay=0.000001)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,30], gamma=0.1)
batch_size = 32
num_epoch = 30

train_dataloader = DataLoader(train_data, batch_size, collate_fn=collate_fn, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size, collate_fn=collate_fn, shuffle=True)

train_loss = []
train_accuracy = []
train_precision = []
train_recall = []
train_f1 = []
test_loss = []
test_accuracy = []
test_precision = []
test_recall = []
test_f1 = []
loss_min = 1.415154
total_time_start = datetime.datetime.now()
for epoch in range(num_epoch):
    model.train()
    train_l, train_a, train_p, train_r, train_f, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    start = datetime.datetime.now()
    for data_x, data_y, batch_seq_len in train_dataloader:
        label = torch.Tensor(data_y).long().to(device)
        hidden, out = model(data_x.to(device),batch_seq_len)
        loss = loss_func(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        prediction = out.argmax(dim=1).data.cpu().numpy()
        label = label.data.cpu().numpy()
        train_l += loss.item()
        train_a += accuracy_score(label, prediction)
        train_p += precision_score(label, prediction, average='weighted')
        train_r += recall_score(label, prediction, average='weighted')
        train_f += f1_score(label, prediction, average='weighted')
        n += 1
    #训练集评价指标
    train_loss.append(train_l/n)
    train_accuracy.append(train_a/n)
    train_precision.append(train_p/n)
    train_recall.append(train_r/n)
    train_f1.append(train_f/n)
    #测试集评价指标
    test_l, test_a, test_p, test_r, test_f = test_evaluate(model, test_dataloader, batch_size, num_epoch)
    test_loss.append(test_l)
    test_accuracy.append(test_a)
    test_precision.append(test_p)
    test_recall.append(test_r)
    test_f1.append(test_f)
    end = datetime.datetime.now()
    print('epoch %d, train: loss %f, accuracy %f, precision %f, recall %f, f1 %f, time %s'% 
          (epoch+1, train_loss[epoch], train_accuracy[epoch], train_precision[epoch], train_recall[epoch], train_f1[epoch], end-start))
    print('          test: loss %f, accuracy %f,  precision %f,  recall %f,  f1 %f'% 
          (test_loss[epoch], test_accuracy[epoch], test_precision[epoch], test_recall[epoch], test_f1[epoch]))
    if test_loss[epoch] < loss_min:
        loss_min = test_loss[epoch]
        torch.save(model.state_dict(), './ex5_0.pt')
        print("save model...")
total_time_end = datetime.datetime.now()
print("疫情微博情绪分类的运行时间：",total_time_end - total_time_start)


# In[18]:


plt.plot(train_loss ,label='train',color='blue')
plt.plot(test_loss ,label='test', color='orange')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title("Loss: batch_size=512, lr=0.01")
plt.legend()
plt.show()


# In[19]:


plt.plot(train_accuracy ,label='train',color='blue')
plt.plot(test_accuracy ,label='test', color='orange')
plt.ylabel("accuracy ")
plt.xlabel("epoch")
plt.title("Accuracy: batch_size=512, lr=0.01")
plt.legend()
plt.show()


# In[20]:


plt.plot(train_precision ,label='train',color='blue')
plt.plot(test_precision ,label='test', color='orange')
plt.ylabel("precision")
plt.xlabel("epoch")
plt.title("Precision: batch_size=512, lr=0.01")
plt.legend()
plt.show()


# In[21]:


plt.plot(train_recall ,label='train',color='blue')
plt.plot(test_recall ,label='test', color='orange')
plt.ylabel("recall")
plt.xlabel("epoch")
plt.title("Recall: batch_size=512, lr=0.01")
plt.legend()
plt.show()


# In[22]:


plt.plot(train_f1 ,label='train',color='blue')
plt.plot(test_f1 ,label='test', color='orange')
plt.ylabel("f1")
plt.xlabel("epoch")
plt.title("F1: batch_size=512, lr=0.01")
plt.legend()
plt.show()


# In[ ]:




