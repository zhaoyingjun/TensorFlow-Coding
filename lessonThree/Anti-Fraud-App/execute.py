import tensorflow as tf
import kmeansModel
import pandas as pd
import numpy as np
import getConfig
import sys
gConfig={}
gConfig=getConfig.get_config(config_file='config.ini')
def read_data(source_file):
    data=pd.read_csv(source_file,encoding='utf-8')
    dataset=data.values
    return dataset

def train():

#设置GPU管理的配置
 config = tf.ConfigProto()
 config.gpu_options.allocator_type = 'BFC'

#读取数据
 dataarray=read_data(gConfig['kmean_train_file'])


 print("Size of data : ",dataarray.shape)
#在会话下进行训练
 with tf.Session(config=config) as sess:

 	model_path=kmeansModel.trainAndSaveModel(dataarray,gConfig['steps'])

def predicts(predict_set):
 	model_path=gConfig['model_path']
 	predict=kmeansModel.predict(model_path,predict_set)
 	return predict

if __name__=='__main__':

    if len(sys.argv) - 1:
        gConfig = getConfig(sys.argv[1])
    else:
        # get configuration from config.ini
        gConfig = getConfig.get_config()
    if gConfig['mode']=='train':
        train()
    elif gConfig['mode']=='server':
        print('Sever Usage:python3 app.py')

 	



