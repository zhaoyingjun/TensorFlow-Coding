import tensorflow as tf
import numpy as np
from six.moves import xrange 
import pandas as pd
import getConfig
gConfig={}
gConfig=getConfig.get_config(config_file='config.ini')


"""
代码结构：

因为本次代码不涉及到复杂的神经网络的搭建和模型的训练，因为就不创建kmeansModel类了，直接定义所需的函数即可。
一共三个函数：
serving_input_receiver_fn():用来获取模型参数，以便于进行模型保存时使用

trainAndSaveModel(input_set,steps)：用来训练模型并将模型保存到指定的文件夹下

predict(export_dir,predict_set)：重新加载模型，并对需要聚类数据进行聚类，返回数据所属类名。


知识点：

tf.estimator 是用来对模型进行训练和评估，包括模型训练，模型保存，模型加载等方法。大家如果感兴趣可以学习一下这个api。

tf.FixedLenFeature 返回的是一个定长的tensor，同时还有一个方法可以返回不定长的,那就是tensortf.FixedLenFeature。

tf.parse_example：是用来将模型解析tensor字典，
比如在features = tf.parse_example(model_placeholder, feature_spec)就是将feature_spec 解析成receiver_tensors张量字典然后返回。


tf.estimator.export.ServingInputReceiver对象是将生成的特征Tensor和占位符组合在一起。
tf.convert_to_tensor：将数组转成Tensor
tf.train.limit_epochs:设置epochs的数量

tf.contrib.factorization.KMeansClustering：这个是Kmeans聚类的类，涵盖了train,predict,cluster_centers，export_savedmodel等方法。有一点要注意，就是其输入的input_fn是一个function，而不是具体的数据。

"""

def serving_input_receiver_fn():
    k=gConfig['encoutlen']
    feature_spec = {"x": tf.FixedLenFeature(dtype=tf.float32, shape=[k])}
    model_placeholder = tf.placeholder(dtype=tf.string,shape=[None],name='input')
    receiver_tensors = {"model_inputs": model_placeholder}
    features = tf.parse_example(model_placeholder, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

def trainAndSaveModel(input_set,steps):


    num_epochs=gConfig['num_epochs']
    num_clusters=gConfig['num_clusters']  
    #定义input_fn函数     
    input_fn = lambda: tf.train.limit_epochs(tf.convert_to_tensor(input_set, dtype=tf.float32), num_epochs=num_epochs)
    #实例化KMeansClustering
    kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters, use_mini_batch=False)
    previous_centers = None
    for _ in xrange(gConfig['steps']):
        kmeans.train(input_fn)#调用train对训练数据进行训练
        centers = kmeans.cluster_centers()#保存质心
        if previous_centers is not None:
            print ("质心变化幅度:", centers - previous_centers)
        previous_centers = centers
        print ("模型评估得分:", kmeans.score(input_fn))
        #将模型保存下来ˇ
        modelPath = kmeans.export_savedmodel(export_dir_base="kmeansMode/",serving_input_receiver_fn=serving_input_receiver_fn)
            
    print("训练完成,model文件存放在：")
    print(modelPath)


"""
知识点：
TFRecords其实是一种二进制文件，虽然它不如其他格式好理解，但是它能更好的利用内存，更方便复制和移动，并且不需要单独的标签文件。
包括前面我们讲到的tf.parse_example以及tf.train.Feature、tf.train.Example都是对这个二进制文件的操作。
一般来说一个Example中包含Features，Features里包含Feature的字典，Feature里包含有一个 FloatList，也可以是ByteList或者Int64List。

"""

def predict(export_dir,predict_set):
          sess = tf.Session()
          import tensorflow.contrib.factorization
          #加载模型
          tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
          from tensorflow.contrib import predictor
          #predictor.from_saved_model是从模型来构造一个预测函数，可以读取之前保存的model，并对输入数据进行聚类。
          predict_fn = predictor.from_saved_model(export_dir)
          inputList=[]
          for test_data in predict_set:
            #以下这块是关于TFRecords的操作，如果大家一时半会理解不了，就可以简单理解为是在准备预测数据特征。
           predictor_input_feature = {
               'x': tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=test_data
                   )
                )
            }

           input_for_predictor = tf.train.Example(
            features=tf.train.Features(
                feature=predictor_input_feature
            )
        )
           #把输入数据转换为String
           serialized_input = input_for_predictor.SerializeToString()
           inputList.append(serialized_input)

          results = predict_fn({"model_inputs": inputList})
          clusterIndices = results['output']

          #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标。

          for i, point in enumerate(predict_set):
             clusterIndex = clusterIndices[i]
             print ('point:', point, 'is in cluster', clusterIndex)
          return clusterIndex





