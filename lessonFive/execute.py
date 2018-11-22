# -*- coding:utf-8 -*-
from time import strftime, localtime 
from datetime import timedelta, date 
import itertools
import tensorflow as tf
import shutil
import os
import pandas as pd
import numpy as np
import getConfig
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#获取配置和数据特征列
gConfig={}

gConfig=getConfig.get_config(config_file='config.ini')

COLUMNS = ['1','2',  '3',  '4',  '5',  '6',  '7',  '8',  '9',  '10', '11', '12', '13', '14', '15', '16', '29', '30', '31']

FEATURES=['1','2',  '3',  '4',  '5',  '6',  '7',  '8',  '9',  '10', '11', '12', '13', '14', '15', '16', '29', '30']

LABEL=['31']

#定义GPU的内存管理算法

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

#定义输入函数
def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels
#定义训练函数
def train():
  # 利用pandas来读取CSV的数据，dataframe格式
  training_set = pd.read_csv(gConfig['training_set'], skipinitialspace=True,skiprows=1, names=COLUMNS)
  test_set = pd.read_csv(gConfig['test_set'], skipinitialspace=True,skiprows=1, names=COLUMNS)

  # 划出和定义特征列


  """

  tf.contrib.layers.real_valued_column：为连续的列元素设置一个实值列

  tf.contrib.learn.DNNClassifier

  classifier.fit

  classifier.evaluate

  """
  feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

 # 构造一个4层，每层105个神经元的全连接的DNN计算图.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,hidden_units=[105, 105, 105, 105 ],dropout=gConfig['keeps'],model_dir=gConfig['model_dir'])
  
  loss_score=1
  # 开始进行训练，知道满足条件后停止
  while loss_score>gConfig['end_loss']:
  
      classifier.fit(input_fn=lambda: input_fn(training_set),steps=100)
      # 测试和评价模型的准确度
      ev = classifier.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
      accuracy_score = ev["accuracy"]
      print("模型准确率: {0:f}".format(accuracy_score))

def init_session(sess,conf='config.ini'):
    global gConfig
    gConfig=getConfig.get_config(conf)
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
    model=tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,hidden_units=[105, 105, 105, 105 ],model_dir=gConfig['model_dir'])
    return sess, model

def predict(sess,predict_set,model):
   
    y=model.predict(input_fn=lambda: input_fn(predict_set))

    predictions = list(y)
    return predictions

if __name__ == "__main__":

    if len(sys.argv) - 1:
        gConfig = getConfig(sys.argv[1])
    else:
        # get configuration from config.ini
        gConfig = getConfig.get_config()
    if gConfig['mode']=='train':
        train()
    elif gConfig['mode']=='server':
        print('Sever Usage:python3 app.py')




        