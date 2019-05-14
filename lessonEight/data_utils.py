
# -*- coding:utf-8 -*-

"""
本次的案例使用的IMDB数据，IMDB提供可以直接加载的API接口，但是为能够实用，特地写了一个数据处理程序。

根据数据的特点，该程序实现如下功能：
读取所有的文件数据，并保存为train test两类有标签的文件以及全部数据集的问津。


"""

import os
import random
import numpy as np
import getConfig
gConfig={}

gConfig=getConfig.get_config(config_file='config.ini')

def read_data(data_path,out_file):

# 读取批量文件后要写入的文件
  with open(out_file, "w") as f:

    # 依次读取根目录下的每一个文件
    for file in os.listdir(data_path):
        file_name = data_path + '/' + file
        filein = open(file_name, "r")
        # 按行读取每个文件中的内容
        for line in filein:
            f.write(line+'\n')
        filein.close()

#下面这5行代码实现从不同的文件中读取分散的数据文件，并保存为根据需要的数据存储文件。
read_data(gConfig['train_pos_data_path'],gConfig['train_pos_data'])

read_data(gConfig['train_neg_data_path'],gConfig['train_neg_data'])

read_data(gConfig['test_pos_data_path'],gConfig['test_pos_data'])
read_data(gConfig['test_neg_data_path'],gConfig['test_neg_data'])
read_data(gConfig['working_directory'],gConfig['all_data'])



