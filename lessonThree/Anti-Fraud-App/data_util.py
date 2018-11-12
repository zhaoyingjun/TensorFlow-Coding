# -*- coding:utf-8 -*-
import math
import os
import random
import getConfig
from tensorflow.python.platform import gfile

import getConfig
gConfig={}
gConfig=getConfig.get_config(config_file='config.ini')
UNK = "__UNK__"
UNK_ID=0
def create_vocabulary(input_file,vocabulary_size,output_file):
    vocabulary = {}
    k=int(vocabulary_size)
    with open(input_file,'r') as f:
         counter = 0
         for line in f:
            counter += 1
            tokens = [word for word in line.split(',')]
            for word in tokens:
                if word in vocabulary:
                   vocabulary[word] += 1
                else:
                   vocabulary[word] = 1
         vocabulary_list = [UNK]+ sorted(vocabulary, key=vocabulary.get, reverse=True)
         
         if len(vocabulary_list) > k:
            vocabulary_list = vocabulary_list[:k]
         print(input_file + " 词汇表大小:", len(vocabulary_list))
         with open(output_file, 'w') as ff:
               for word in vocabulary_list:
                   ff.write(word + "\n")

def convert_to_vector(input_file, vocabulary_file, output_file):
	tmp_vocab = []
	with open(vocabulary_file, "r") as f:#读取字典文件的数据，生成一个dict，也就是键值对的字典
		tmp_vocab.extend(f.readlines())
	tmp_vocab = [line.strip() for line in tmp_vocab]
	vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])#将vocabulary_file中的键值对互换，因为在字典文件里是按照{123：好}这种格式存储的，我们需要换成{好：123}格式

	output_f = open(output_file, 'w')
	with open(input_file, 'r') as f:
		for line in f:
			line_vec = []
			for words in line.split(','):
				line_vec.append(vocab.get(words, UNK_ID))#获取words的对应编码，如果找不到就返回UNK_ID
			output_f.write(" ".join([str(num) for num in line_vec]) + "\n")#将input_file里的中文字符通过查字典的方式，替换成对应的key，并保存在output_file
	output_f.close()


def prepare_custom_data(working_directory,input_file,vocabulary_size,):
    # 生成字典的路径，encoder和decoder的字典是分开的
    vocab_path = os.path.join(working_directory, "vocab%d" % vocabulary_size)
   
    create_vocabulary(input_file,vocabulary_size,vocab_path)
   
    # 将训练数据集的中文用字典进行替换
    ids_path = input_file + (".ids%d" % vocabulary_size)
    data=convert_to_vector(input_file,vocab_path,ids_path)

    return data

#def main():

  #  working_directory=gConfig['working_directory']
  #  input_file=gConfig['input_file']
  #  vocabulary_size=gConfig['vocabulary_size']

    prepare_custom_data(working_directory,input_file,vocabulary_size)


#if __name__ == '__main__':
  #  main()


