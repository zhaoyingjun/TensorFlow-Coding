"""
小象学院TensorFlow高级实践第一课：TensorFlow编程基础入门
"""
import tensorflow as tf
import numpy as np
import pickle
import getConfig


gConfig={}
gConfig=getConfig.get_config(config_file='config.ini')

"""
cnnModel的代码结构：
1、定义类
2、参数初始化，__init__函数
3、create_conv_layer定义卷积层函数
4、create_CNN 定义卷积神经网络,可以理解为cnnfouction
5、dropout_flatten_layer，dropout函数用于CNN网络的dropout
我们在这里补充一下dropoupt的知识：
简单来说dropout解决了两个问题：一个是复杂的网络计算量非常大，需要较长的时间计算，一个复杂的网络会造成过拟合，网络的泛化性堪忧。
那么dropout如何做的？
a、对于一个有N个节点的神经网络，有了dropout之后就可以看做是2^n个模型的集合了，但此时要训练的参数数目却是不变的，这样的计算复杂度就指数级的下降
b、dropout它强迫一个神经单元，和随机挑选出来的其他神经单元共同工作，达到好的效果。消除减弱了神经元节点间的联合适应性，增强了泛化能力。
6、fc_layer:全连接层
7、step（）：执行训练和预测任务

"""


class cnnModel(object):
    def __init__(self,percent,learning_rate,learning_rate_decay_factor):
        self.percent=percent
        self.learning_rate=tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)

        def create_conv_layer(input_data, filter_size, num_filters):
            filters = tf.Variable(tf.truncated_normal(shape=(
            filter_size, filter_size, tf.cast(input_data.shape[-1], dtype=tf.int32), num_filters),
                                                                     stddev=0.05))
            print("Size of conv filters bank : ", filters.shape)

            conv_layer = tf.nn.conv2d(input=input_data,
                                              filter=filters,
                                              strides=[1, 1, 1, 1],
                                              padding="VALID")
            print("Size of conv result : ", conv_layer.shape)

            return filters, conv_layer

        def create_CNN(input_data, num_classes, keep_prop):
            filters1, conv_layer1 = create_conv_layer(input_data=input_data, filter_size=5, num_filters=4)
            relu_layer1 = tf.nn.relu(conv_layer1)
            print("Size of relu1 result : ", relu_layer1.shape)
            max_pooling_layer1 = tf.nn.max_pool(value=relu_layer1,
                                                        ksize=[1, 2, 2, 1],
                                                        strides=[1, 1, 1, 1],
                                                        padding="VALID")
            print("Size of maxpool1 result : ", max_pooling_layer1.shape)

            filters2, conv_layer2 = create_conv_layer(input_data=max_pooling_layer1, filter_size=7, num_filters=3)
            relu_layer2 = tf.nn.relu(conv_layer2)
            print("Size of relu2 result : ", relu_layer2.shape)
            max_pooling_layer2 = tf.nn.max_pool(value=relu_layer2,
                                                        ksize=[1, 2, 2, 1],
                                                        strides=[1, 1, 1, 1],
                                                        padding="VALID")
            print("Size of maxpool2 result : ", max_pooling_layer2.shape)

            # Conv layer with 2 filters and a filter sisze of 5x5.
            filters3, conv_layer3 = create_conv_layer(input_data=max_pooling_layer2, filter_size=5, num_filters=2)
            relu_layer3 = tf.nn.relu(conv_layer3)
            print("Size of relu3 result : ", relu_layer3.shape)
            max_pooling_layer3 = tf.nn.max_pool(value=relu_layer3,
                                                        ksize=[1, 2, 2, 1],
                                                        strides=[1, 1, 1, 1],
                                                        padding="VALID")
            print("Size of maxpool3 result : ", max_pooling_layer3.shape)

            # Adding dropout layer before the fully connected layers to avoid overfitting.
            flattened_layer = dropout_flatten_layer(previous_layer=max_pooling_layer3, keep_prop=keep_prop)

            # First fully connected (FC) layer. It accepts the result of the dropout layer after being flattened (1D).
            fc_resultl = fc_layer(flattened_layer=flattened_layer,
                                  num_inputs=flattened_layer.get_shape()[1:].num_elements(),
                                  num_outputs=200)
            # Second fully connected layer accepting the output of the previous fully connected layer. Number of outputs is equal to the number of dataset classes.
            fc_result2 = fc_layer(flattened_layer=fc_resultl, num_inputs=fc_resultl.get_shape()[1:].num_elements(),
                                  num_outputs=num_classes)
            print("Fully connected layer results : ", fc_result2)
            return fc_result2  # Returning the result of the last FC layer.

        def dropout_flatten_layer(previous_layer, keep_prop):

            dropout = tf.nn.dropout(x=previous_layer, keep_prob=keep_prop)
            num_features = dropout.get_shape()[1:].num_elements()
            layer = tf.reshape(previous_layer, shape=(-1, num_features))  # Flattening the results.
            return layer
        #tf.truncated_normal这是一个正态分布函数，可以符合正态分布的数
        def fc_layer(flattened_layer, num_inputs, num_outputs):
            # 根据输入inputs和outputs的数量来生成weights的矩阵
            fc_weights = tf.Variable(tf.truncated_normal(shape=(num_inputs, num_outputs),stddev=0.05))
            # 将网络矩阵与权重相乘，随着输入输出的调整来调整每一个网络输入的权重.
            fc_resultl = tf.matmul(flattened_layer, fc_weights)
            return fc_resultl
        self.data_tensor=tf.placeholder(tf.float32,shape=[None,gConfig['im_dim'], gConfig['im_dim'],gConfig['num_channels']],name='data_tensor')
        self.label_tensor=tf.placeholder(tf.float32,shape=[None],name='label_tensor')
        keep_prop=tf.Variable(initial_value=0.5,name="keep_prop")
        fc_result=create_CNN(input_data=self.data_tensor,num_classes=gConfig['num_dataset_classes'],keep_prop=gConfig['keeps'])
        self.softmax_propabilities=tf.nn.softmax(fc_result,name="softmax_probs")
        self.softmax_predictions=tf.argmax(self.softmax_propabilities,axis=1)
        #计算交叉熵
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=tf.reduce_max(input_tensor=self.softmax_propabilities,reduction_indices=[1]),
                                                              labels=self.label_tensor)
        #计算损失
        cost=tf.reduce_mean(cross_entropy)
        self.ops=tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        #保存所有变量的值
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self,sess,shuffled_data,shuffled_labels,graph,forward_only=None):

        keep_prop = tf.placeholder(tf.float32)
        gConfig=getConfig.get_config(config_file='config.ini')
        #是否只进行正向传播，及正向传播是进行预测，反向传播是进行训练
        if forward_only:
            keep_prop = graph.get_tensor_by_name(name="keep_prop:0")
            #softmax_propabilities = graph.get_tensor_by_name(name="softmax_probs:0")
            #softmax_predictions = tf.argmax(softmax_propabilities, axis=1)
            #data_tensor = graph.get_tensor_by_name(name="data_tensor:0")
            dataset_array = np.random.rand(1, 32, 32, 3)
            dataset_array[0,:,:,:] = shuffled_data
            feed_dict_test={self.data_tensor:dataset_array,keep_prop:1.0
            }
            softmax_propabilities_, softmax_predictions_ = sess.run([self.softmax_propabilities, self.softmax_predictions],
                                                         feed_dict=feed_dict_test)
            file=gConfig['dataset_path'] + "batches.meta"
            patch_bin_file = open(file, 'rb')
            label_names_dict = pickle.load(patch_bin_file)
            print(label_names_dict)
            dataset_label_names = label_names_dict["label_names"]
            return dataset_label_names[softmax_predictions_[0]]
        else:
            cnn_feed_dict = {self.data_tensor: shuffled_data, self.label_tensor: shuffled_labels, keep_prop: gConfig['keeps']}
            softmax_predictions_, _ = sess.run([self.softmax_predictions, self.ops],feed_dict=cnn_feed_dict)
            # 计算预测准确的数量
        
            correct = np.array(np.where(softmax_predictions_ == shuffled_labels))
            print(correct)

            accuracy = correct.size/(self.percent*gConfig['dataset_size']/100)
            return accuracy #输出准确率

        """
        准确率（accuracy）： 正确预测占全部样本的比例

        精准率（precision）：正确预测为正占全部预测为正的比例

        召回率（recall）： 正确预测为正占全部正样本的比例
        
        
        """





















