# -*- coding:utf-8 -*-
import math
import os
import random
import getConfig
from tensorflow.python.platform import gfile
import re
#不管是中文还是英文都是人类能够识别的语言，对于计算机来说需要将人类的语言转换为计算机能够识别和计算的数字编码
#prepareData就是来完成这一过程的，大概需要做以下几步来完成
#1、生成字典，
#2、用字典将数据集的中文汉字替换成字典对应的编码
#PAD是一个填充标志，换句话来说就是为了让每个桶里的数据是一样长的，这样方便计算。
#GO是作为一句话的开始标志
#EOC是用为一句话的结束标志
#UNK是为了标记不在字典里的数据，这个非常关键，因为字典里的字总是有限的，加上UNK可以将一些生僻的字标记出来

PAD = "__PAD__"
GO = "__GO__"
EOS = "__EOS__"  # 对话结束
UNK = "__UNK__"  # 标记未出现在词汇表中的字符
START_VOCABULART = [PAD, GO, EOS, UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
# 定义字典生成函数
#生成字典的原理其实很简单，就是统计所有训练数据中的词频，然后按照词频进行排序，每个词在训练集中出现次数就是其对应的编码
#关于字典这一块，其实就是一个一个约定，不一定用词频的方式，也可以按照新华字典的编号。
#知识点：函数定义、在函数中函数的调用是不需要声明的、字典类型
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
          # 取前20000个常用汉字
         if len(vocabulary_list) > k:
            vocabulary_list = vocabulary_list[:k]
         print(input_file + " 词汇表大小:", len(vocabulary_list))
         with open(output_file, 'w') as ff:
               for word in vocabulary_list:
                   ff.write(word + "\n")

#在生成字典之后，我们就需要将我们之前训练集的中文全部用字典进行替换
#知识点：list的append和extend，dict的get操作、文件的写入操作

# 把对话字符串转为向量形式
def convert_to_vector(input_file, vocabulary_file, output_file):
	print('对话转向量...')
	tmp_vocab = []
	with open(vocabulary_file, "r") as f:#读取字典文件的数据，生成一个dict，也就是键值对的字典
		tmp_vocab.extend(f.readlines())
	tmp_vocab = [line.strip() for line in tmp_vocab]
	vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])#将vocabulary_file中的键值对互换，因为在字典文件里是按照{123：好}这种格式存储的，我们需要换成{好：123}格式

	output_f = open(output_file, 'w')
	with open(input_file, 'r') as f:
		for line in f:
			line_vec = []
			for words in line.split():
				line_vec.append(vocab.get(words, UNK_ID))#获取words的对应编码，如果找不到就返回UNK_ID
			output_f.write(" ".join([str(num) for num in line_vec]) + "\n")#将input_file里的中文字符通过查字典的方式，替换成对应的key，并保存在output_file
	output_f.close()

#然后我们需要定义一个接口，供其他方法调用，就是将以上的两个函数的处理过程封装在一个接口里。

def prepare_custom_data(working_directory, train_enc, train_dec, test_enc, test_dec, enc_vocabulary_size, dec_vocabulary_size, tokenizer=None):

    # 生成字典的路径，encoder和decoder的字典是分开的
    enc_vocab_path = os.path.join(working_directory, "vocab%d.enc" % enc_vocabulary_size)
    dec_vocab_path = os.path.join(working_directory, "vocab%d.dec" % dec_vocabulary_size)
    #生成字典文件
    create_vocabulary(train_enc,enc_vocabulary_size,enc_vocab_path)
    create_vocabulary(train_dec,dec_vocabulary_size,dec_vocab_path)
   
    # 将训练数据集的中文用字典进行替换
    enc_train_ids_path = train_enc + (".ids%d" % enc_vocabulary_size)
    dec_train_ids_path = train_dec + (".ids%d" % dec_vocabulary_size)
    convert_to_vector(train_enc, enc_vocab_path, enc_train_ids_path)
    convert_to_vector(train_dec, dec_vocab_path, dec_train_ids_path)
 

    # 将测试数据集的中文用字典进行替换
    enc_dev_ids_path = test_enc + (".ids%d" % enc_vocabulary_size)
    dec_dev_ids_path = test_dec + (".ids%d" % dec_vocabulary_size)
    convert_to_vector(test_enc, enc_vocab_path, enc_dev_ids_path)
    convert_to_vector(test_dec, dec_vocab_path, dec_dev_ids_path)

    return (enc_train_ids_path, dec_train_ids_path, enc_dev_ids_path, dec_dev_ids_path, enc_vocab_path, dec_vocab_path)

# 用于语句切割的正则表达
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def basic_tokenizer(sentence):
  #将一个语句中的字符切割成一个list，这样是为了下一步进行向量化训练
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def sentence_to_token_ids(sentence, vocabulary, normalize_digits=True):#将输入语句从中文字符转换成数字符号

  words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
 # # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]

def initialize_vocabulary(vocabulary_path):#初始化字典，这里的操作与上面的48行的的作用是一样的，是对调字典中的key-value
   if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with open(vocabulary_path, "r") as f:
      rev_vocab.extend(f.readlines())
      rev_vocab = [line.strip() for line in rev_vocab]
      vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
   else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

