# coding=utf-8

import pandas as pd
import numpy as np
import getConfig

import os
import random
#常规的获取配置信息
gConfig = {}

gConfig=getConfig.get_config()

conv_path = gConfig['resource_data']

#定义csv的列名，用于区分特征列和标示列

COLUMNS = ['1','2','3','4','5','6','7','8','9','10','11','12','13', '14', '15', '16', '29', '30', '31']

resource_data=pd.read_csv(conv_path,skipinitialspace=True,skiprows=1, names=COLUMNS,low_memory=False)
#利用pd 的dataframe特性进行缺省值补充
"""
知识点：
缺省值补全：
df.fillna(0)
df.fillna('missing')
df.fillna(method='pad')

df.fillna(method='bfill',limit=1)

df.fillna(df.mean()）

dataframe.to_csv


"""
resource_data=resource_data.fillna(method='bfill')

def sample_test_data(resource_data,TESTSET_SIZE):

	    test_index = random.sample([i for i in range(len(resource_data))],TESTSET_SIZE)

	    train_data=[]
	    test_data=[]
 
	    for i in range(len(resource_data)):
	        if i in test_index:
	          #print(resource_data.loc[i])
	          test_data.append(resource_data.loc[i])
            
	        else:
	           train_data.append(resource_data.loc[i])
	        if i % 1000 == 0:
	           print(len(range(len(resource_data))), '处理进度：', i)
	    train_data_tocsv=pd.DataFrame(train_data)

	    test_data_tocsv=pd.DataFrame(test_data)

	    #train_data_tocsv=(train_data_tocsv - train_data_tocsv.min())/(train_data_tocsv.max() - train_data_tocsv.min())
	    #test_data_tocsv=(test_data_tocsv - test_data_tocsv.min()) / (test_data_tocsv.max() - test_data_tocsv.min())

	    train_data_path=gConfig['training_set']

	    test_data_path=gConfig['test_set']

	    train_data_tocsv.to_csv(train_data_path)

	    test_data_tocsv.to_csv(test_data_path)
 

sample_test_data(resource_data,10000)





