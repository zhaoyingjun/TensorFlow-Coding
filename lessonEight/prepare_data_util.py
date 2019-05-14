
# -*- coding:utf-8 -*-

"""
这个程序的作用是，将之前处理好的文字数据，转换成词向量，并根据需要保存为通用的npz的格式。

这里和我们lessonTwo的prepareData的内容是类似的，采用的词典生成方式也是一样的。


代码结构：


1、生成词典：create_vocabulary

2、词向量转换：convert_to_vector

3、词向量的存储：



"""

import os
import random
import numpy as np
import getConfig
import re
gConfig={}

gConfig=getConfig.get_config(config_file='config.ini')

UNK = "__UNK__"  # 标记未出现在词汇表中的字符
START_VOCABULART = [UNK]
UNK_ID = 3
# 定义字典生成函数
#生成字典的原理其实很简单，就是统计所有训练数据中的词频，然后按照词频进行排序，每个词在训练集中出现次数就是其对应的编码
#知识点：函数定义、在函数中函数的调用是不需要声明的、字典类型

"""
词频词典的创建：
1、读取所有的文字
2、统计每个文字的出现次数
3、排序
4、取值保存

"""
def create_vocabulary(input_file,vocabulary_size,output_file):
    vocabulary = {}
    k=int(vocabulary_size)
    with open(input_file,'r') as f:
         counter = 0
         for line in f:
            counter += 1
            tokens = [word for word in line.split()]
            for word in tokens:
                if word in vocabulary:
                   vocabulary[word] += 1
                else:
                   vocabulary[word] = 1
         vocabulary_list = START_VOCABULART + sorted(vocabulary, key=vocabulary.get, reverse=True)
          # 根据配置，取vocabulary_size大小的词典
         if len(vocabulary_list) > k:
            vocabulary_list = vocabulary_list[:k]
        #将生成的词典保存到文件中
         print(input_file + " 词汇表大小:", len(vocabulary_list))
         with open(output_file, 'w') as ff:
               for word in vocabulary_list:
                   ff.write(word + "\n")

#在生成字典之后，我们就需要将我们之前训练集的文字全部用字典进行替换
#知识点：list的append和extend，dict的get操作、文件的写入操作

# 把对话字符串转为向量形式

"""
1、遍历文件
2、找到一个字 然后在词典出来，然后做替换
3、保存文件



"""
def convert_to_vector(input_file, vocabulary_file, output_file):
    print('文字转向量...')
    tmp_vocab = []
    with open(vocabulary_file, "r") as f:#读取字典文件的数据，生成一个dict，也就是键值对的字典
         tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])#将vocabulary_file中的键值对互换，因为在字典文件里是按照{123：好}这种格式存储的，我们需要换成{好：123}格式

    output_f = open(output_file, 'w')
    with open(input_file, 'r') as f:
        line_out=[]
        for line in f:
            line_vec = []
            for words in line.split():
                line_vec.append(vocab.get(words, UNK_ID))#获取words的对应编码，如果找不到就返回UNK_ID
            output_f.write(" ".join([str(num) for num in line_vec]) + "\n")#将input_file里的中文字符通过查字典的方式，替换成对应的key，并保存在output_file
            #print(line_vec)
            line_out.append(line_vec)
        output_f.close()
        return line_out




def prepare_custom_data(working_directory,train_pos,train_neg,test_pos,test_neg,all_data,vocabulary_size):

    # 生成字典的路径，encoder和decoder的字典是分开的
    vocab_path = os.path.join(working_directory, "vocab%d.txt" % vocabulary_size)
    
    #生成字典文件
    create_vocabulary(all_data,vocabulary_size,vocab_path)
    # 将训练数据集的中文用字典进行替换
    pos_train_ids_path = train_pos + (".ids%d" % vocabulary_size)
    neg_train_ids_path = train_neg + (".ids%d" % vocabulary_size)
    train_pos=convert_to_vector(train_pos, vocab_path, pos_train_ids_path)
    train_neg=convert_to_vector(train_neg, vocab_path, neg_train_ids_path)
 

    # 将测试数据集的中文用字典进行替换
    pos_test_ids_path = test_pos + (".ids%d" % vocabulary_size)
    neg_test_ids_path = test_neg + (".ids%d" % vocabulary_size)
    test_pos=convert_to_vector(test_pos, vocab_path, pos_test_ids_path)
    test_neg=convert_to_vector(test_neg, vocab_path, neg_test_ids_path)
    return train_pos,train_neg,test_pos,test_neg
train_pos,train_neg,test_pos,test_neg=prepare_custom_data(gConfig['working_directory'],gConfig['train_pos_data'],gConfig['train_neg_data'],gConfig['test_pos_data'],gConfig['test_neg_data'],gConfig['all_data'],gConfig['vocabulary_size'])

y_trian=[]
y_test=[]

for i in range(len(train_pos)):
    y_trian.append(0)
for i in range(len(train_neg)):
    y_trian.append(1)

for i in range(len(test_pos)):
   y_test.append(0)
for i in range(len(test_neg)):
    y_test.append(1)


x_train=np.concatenate((train_pos,train_neg),axis=0)
x_test=np.concatenate((test_pos,test_neg),axis=0)

np.savez("data/imdb.npz",x_train,y_trian,x_test,y_test)