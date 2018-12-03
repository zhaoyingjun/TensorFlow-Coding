import  tensorflow as tf
import numpy as np
import vggModel
import os
import pickle
import time
import getConfig
import sys
import random
gConfig = {}

def read_data(dataset_path, im_dim, num_channels,num_files,images_per_file):
        files_names = os.listdir(dataset_path)
        print(files_names)
          # 获取训练集中训练文件的名称
        """
        在CIFAR10中已经为我们标注和准备好了数据，一时找不到合适的高质量的标注训练集，我们就是使用CIFAR10的来作为我们的训练集。
        在训练集中一共有50000个训练样本，放到5个二进制文件中心，每个样本有3072个像素点，是32*3维度的
        """
        #创建空的多维数组用于存放图片二进制数据
        dataset_array = np.zeros(shape=(num_files * images_per_file, im_dim, im_dim, num_channels))
        # 创建空的数组用于存放图片的标注信息
        dataset_labels = np.zeros(shape=(num_files * images_per_file), dtype=np.uint8)
        index = 0
        #从训练集中读取二进制数据并将其维度转换成224*224*3
        for file_name in files_names:

            if file_name[0:len(file_name)-1] == "data_batch_":
                print("正在处理数据 : ", file_name)
                data_dict = unpickle_patch(dataset_path + file_name)
                images_data = data_dict[b"data"]
                # 格式转换为224x224x3 shape.
                print(images_data.shape)
                images_data_reshaped = np.reshape(images_data,
                                                     newshape=(len(images_data), im_dim, im_dim, num_channels))
                # 将维度转换后的图片数据存入指定数组内
                dataset_array[index * images_per_file:(index + 1) * images_per_file, :, :, :] = images_data_reshaped
                #  将维度转换后的标注数据存入指定数组内
                dataset_labels[index * images_per_file:(index + 1) * images_per_file] = data_dict[b"labels"]
                index = index + 1
        return dataset_array, dataset_labels  # 返回数据

def unpickle_patch(file):
    #打开文件，读取二进制文件，返回读取到的数据
    patch_bin_file = open(file, 'rb')
    patch_dict = pickle.load(patch_bin_file, encoding='bytes')#Loading the details of the binary file into a dictionary.
    return patch_dict

def create_model(session,forward_only):

    #用vggModel实例化一个对象model
    model=vggModel.vggModel(gConfig['percent'],gConfig['learning_rate'],gConfig['learning_rate_decay_factor'])
    if 'pretrained_model'in gConfig:
        model.saver.restore(session,gConfig['pretrained_model'])
        return model
    ckpt=tf.train.get_checkpoint_state(gConfig['working_directory'])
    #判断是否已经有Model文件存在，如果model文件存在则加载原来的model并在原来的moldel继续训练，如果不存在则新建model相关文件
    if ckpt and ckpt.model_checkpoint_path:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()

        return model,graph

    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        graph = tf.get_default_graph()
        return model,graph
#获取批量处理数据，考虑到配置不同，如果没有GPU建议将percent调小一点，即将训练集调小
def get_batch(data,labels,percent):
    num_elements = np.uint32(percent * data.shape[0] / 100)
    #之所以没有采用tf自带的tf.train.batch是因为那个接口对内存的消耗太大了，我们采用这种随机取index的方法简单又实用。
    n=np.random.randint(len(labels),size=num_elements)
    shuffle_datas=[]
    shuffled_labels=[]
    for i in n:
        shuffle_datas.append(data[i,:,:,:])
        shuffled_labels.append(labels[i])
    return shuffle_datas,shuffled_labels

#定义训练函数
def train():
    """使用BFC内存管理管理算法，tf.ConfigProto()用于GPU的管理，可以控制GPU的使用率
    #allow growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # per_process_gpu_memory_fraction
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config=tf.ConfigProto(gpu_options=gpu_options)
       关于BFC算法：
      将内存分块管理，按块进行空间分配和释放。
     通过split操作将大内存块分解成用户需要的小内存块。
     通过merge操作合并小的内存块，做到内存碎片回收
     通过bin这个抽象数据结构实现对空闲块高效管理。
    """

    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'

    #读取训练集的数据到内存中
    dataset_array, dataset_labels = read_data(dataset_path=gConfig['dataset_path'], im_dim=gConfig['im_dim'],
                                            num_channels=gConfig['num_channels'],num_files=gConfig['num_files'],images_per_file=gConfig['images_per_file'])
    #打印训练数据的维度
    print("Size of data : ", dataset_array.shape)
    #读取测试数据到内存中
    dataset_array_test, dataset_labels_test = read_data(dataset_path=gConfig['dataset_test'], im_dim=gConfig['im_dim'], num_channels=gConfig['num_channels'],num_files=1,images_per_file=gConfig['images_per_file'])
    #打印测试数据的维度
    print("Size of data : ", dataset_array_test.shape)

    with tf.Session(config=config) as sess:
        model,_=create_model(sess,False)

        step_time, accuracy = 0.0, 0.0
        current_step = 0
        previous_correct = []
        # 开始训练循环，这里没有设置结束条件，当学习率下降到为0时停止
        while model.learning_rate.eval()>gConfig['end_learning_rate']:

            shuffled_datas, shuffled_labels = get_batch(data=dataset_array, labels=dataset_labels,
                                                             percent=gConfig['percent'])
            shuffled_datas_test, shuffled_labels_test = get_batch(data=dataset_array_test, labels=dataset_labels_test,
                                                             percent=gConfig['percent']*5)

            start_time = time.time()
            step_correct=model.step(sess,shuffled_datas,shuffled_labels,False)
            step_time += (time.time() - start_time) / gConfig['steps_per_checkpoint']
            accuracy += step_correct / gConfig['steps_per_checkpoint']
            current_step += 1

            # 达到一个训练模型保存点后，将模型保存下来，并打印出这个保存点的平均准确率
            if current_step % gConfig['steps_per_checkpoint'] == 0:
                #如果超过三次预测正确率没有升高则改变学习率
                if len(previous_correct) > 2 and accuracy < min(previous_correct[-5:]):
                    sess.run(model.learning_rate_decay_op)
                previous_correct.append(accuracy)
                checkpoint_path = os.path.join(gConfig['working_directory'], "vgg.ckpt")
                #saver=tf.train.Saver()
                model.saver.save(sess, checkpoint_path,global_step=model.global_step)

            
                #以下为增加模型在测试集上的准确率测试
                graph = tf.get_default_graph()
                #以下获取默认计算图的参数
                softmax_propabilities = graph.get_tensor_by_name(name="softmax_probs:0")
                softmax_predictions = tf.argmax(softmax_propabilities, axis=1)
                data_tensor = graph.get_tensor_by_name(name="data_tensor:0")
                label_tensor = graph.get_tensor_by_name(name="label_tensor:0")
                keep_prob = graph.get_tensor_by_name(name="keep_prob:0")

                feed_dict_testing = {data_tensor: shuffled_datas_test,
                    label_tensor: shuffled_labels_test,
                     keep_prob: 1.0}

                softmax_propabilities_, softmax_predictions_ = sess.run([softmax_propabilities, softmax_predictions],
                                                      feed_dict=feed_dict_testing)
                
                correct = np.array(np.where(softmax_predictions_ == shuffled_labels_test))
                correct = correct.size
                #打印出模型在训练集上的准确率
                print("在", str(gConfig['percent'] *gConfig['dataset_size']/100),"个训练集上训练的准确率", ' : ', accuracy)
                print("学习率 %.4f 每步耗时 %.2f  " % ( model.learning_rate.eval(),step_time))
                step_time, accuracy = 0.0,0.0
                
                #打印出模型在测试集上的准确率
                print("模型在测试集上的准确率为 : ", correct/(gConfig['percent']*gConfig['dataset_size']/100))

                sys.stdout.flush()   

def init_session(sess,conf='config.ini'):
    global gConfig
    gConfig=getConfig.get_config(conf)
    model,graph=create_model(sess,True)
    return sess, model,graph

def predict_line(sess,model,img,graph):
    predict_name=model.step(sess,img,img,graph,True)
    return predict_name

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