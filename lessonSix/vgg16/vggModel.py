"""

"""
import tensorflow as tf
import numpy as np
import pickle
import getConfig
from collections import Counter

import tensorflow.contrib.slim as slim


gConfig={}
gConfig=getConfig.get_config(config_file='config.ini')

"""
vgg16使用tf.contrin.slim大大简化了代码，如果按照传统的写法写vgg16那可能要写死人，现在我们
只要30行以内的代码就可以完成一个vgg16 的构建。

因此这里我们重点将vgg16的代码结构


知识点：
tf.truncated_normal_initializer

l2_regularizer


"""

class vggModel(object):
    def __init__(self,percent,learning_rate,learning_rate_decay_factor):
        self.percent=percent
        self.learning_rate=tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        def vgg16Model(input_data, num_classes,keep_prob):
                
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
                         #第一段卷积堆叠，两层堆叠，conv3-64的意义是3*3的卷积核一共64个
                         net = slim.repeat(input_data, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                         net = slim.max_pool2d(net, [2, 2], scope='pool1')
                         #第二段卷积核堆叠，两层堆叠，128个3*3的卷积核，注意这里和感受野的区别
                         net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                         net = slim.max_pool2d(net, [2, 2], scope='pool2')
                         #第三段卷积核堆叠，三层堆叠，256个3*3的卷积核，注意这里和感受野的区别
                         net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                         net = slim.max_pool2d(net, [2, 2], scope='pool3')
                         #第四段卷积核堆叠，三层堆叠，512个3*3的卷积核，注意这里和感受野的区别
                         net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                         net = slim.max_pool2d(net, [2, 2], scope='pool4')
                         #第五段卷积核堆叠，三层堆叠，512个3*3的卷积核，注意这里和感受野的区别
                         net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                         net = slim.max_pool2d(net, [2, 2], scope='pool5')
                         #Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
                         net = slim.flatten(net, scope='flat')
                         #全连接层
                         net = slim.fully_connected(net, 4096, scope='fc1')
                         net = slim.dropout(net, keep_prob=keep_prob, scope='dropt1')
                          #全连接层
                         net = slim.fully_connected(net, 4096, scope='fc2')
                         net = slim.dropout(net, keep_prob=keep_prob, scope='dropt2')
                          #全连接层
                         net = slim.fully_connected(net, num_classes, scope='fc3')
                         net = slim.softmax(net, scope='net')

                return net

        #这块内容和第一课的内容是一样的
        batch_size=gConfig['percent']*gConfig['dataset_size']/100
        self.data_tensor=tf.placeholder(tf.float32,shape=[batch_size,gConfig['im_dim'], gConfig['im_dim'],gConfig['num_channels']],name='data_tensor')
        self.label_tensor=tf.placeholder(tf.int32,shape=[batch_size],name='label_tensor')
        keep_prob=tf.Variable(initial_value=0.5,name="keep_prob")
        self.fc_result=vgg16Model(input_data=self.data_tensor,num_classes=gConfig['num_dataset_classes'],keep_prob=gConfig['keeps'])
        self.softmax_propabilities=tf.nn.softmax(self.fc_result,name="softmax_probs")
        self.softmax_predictions=tf.argmax(self.softmax_propabilities,axis=1)
            
        #将label变成one-hot编码，因为softmax_propabilities是一个数组，是10个概率，每个概率代表着预测结果属于其index类的概率，为了计算交叉熵，我们需要把label也转换成一个数组
        self.label_tensor=tf.one_hot(int(batch_size),self.label_tensor,10)
        #计算交叉熵
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.softmax_propabilities,
                                                              labels=self.label_tensor)
        cost=tf.reduce_mean(cross_entropy)

        self.ops=tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost,global_step=self.global_step)
        #保存所有变量的
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self,sess,shuffled_data,shuffled_labels,graph,forward_only=None):
        
        keep_prob = tf.placeholder(tf.float32)
        gConfig=getConfig.get_config(config_file='config.ini')
        #是否只进行正向传播，及正向传播是进行预测，反向传播是进行训练
        if forward_only:
            keep_prob = graph.get_tensor_by_name(name="keep_prob:0")
            data_tensor=graph.get_tensor_by_name(name="data_tensor:0")
            k_size=gConfig['percent']*gConfig['dataset_size']/100
            dataset_array = np.random.rand(int(k_size), 32, 32, 3)
            dataset_array[0,:,:,:] = shuffled_data
            print(shuffled_data)
            print(dataset_array[0])
            print(dataset_array)
            feed_dict_test={data_tensor:dataset_array,keep_prob:1.0
            }
            softmax_propabilities_, softmax_predictions_ = sess.run([self.softmax_propabilities, self.softmax_predictions],
                                                         feed_dict=feed_dict_test)

            file=gConfig['dataset_path'] + "batches.meta"
            patch_bin_file = open(file, 'rb')
            label_names_dict = pickle.load(patch_bin_file)
            print(label_names_dict)
            print(softmax_predictions_[0])
            print(softmax_predictions_)
            print(Counter(softmax_predictions_).most_common(1))
            
            #k=Counter(softmax_predictions_).most_common(1)
            #print(k)
            dataset_label_names = label_names_dict["label_names"]
            return dataset_label_names[softmax_predictions_[0]]
        else:   

        	cnn_feed_dict = {self.data_tensor: shuffled_data, self.label_tensor: shuffled_labels, keep_prob: gConfig['keeps']}
        	softmax_predictions_, _ = sess.run([self.softmax_predictions, self.ops],feed_dict=cnn_feed_dict)
            # 统计预测争取的数量
        	correct = np.array(np.where(softmax_predictions_ == shuffled_labels))

        	accuracy = correct.size/(self.percent*gConfig['dataset_size']/100)
        	return accuracy #输出准确率

        """
        准确率（accuracy）： 正确预测占全部样本的比例

        精准率（precision）：正确预测为正占全部预测为正的比例

        召回率（recall）： 正确预测为正占全部正样本的比例
        
        
        """





















