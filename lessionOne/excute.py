import  tensorflow as tf
import numpy as np
import cnnModel
import configparser
import os
import pickle
import time
import getConfig
import sys
gConfig = {}
#num_datatset_classes = 10
#im_dim = 32
#num_channels = 3
#patches_dir="/Users/zhaoyingjun/Learning/tensorflow_coding/data/"
#gConfig= getConfig(config_file='config.ini')

def read_data(dataset_path, im_dim, num_channels,num_files,images_per_file):
        #num_files = 5  # Number of training binary files in the CIFAR10 dataset.
        #images_per_file = 10000  # Number of samples withing each binary file.
        files_names = os.listdir(dataset_path)  # Listing the binary files in the dataset path.
        """
        Creating an empty array to hold the entire training data after being reshaped.
        The dataset has 5 binary files holding the data. Each binary file has 10,000 samples. Total number of samples in the dataset is 5*10,000=50,000.
        Each sample has a total of 3,072 pixels. These pixels are reshaped to form a RGB image of shape 32x32x3.
        Finally, the entire dataset has 50,000 samples and each sample of shape 32x32x3 (50,000x32x32x3).
        """
        dataset_array = np.zeros(shape=(num_files * images_per_file, im_dim, im_dim, num_channels))
        # Creating an empty array to hold the labels of each input sample. Its size is 50,000 to hold the label of each sample in the dataset.
        dataset_labels = np.zeros(shape=(num_files * images_per_file), dtype=np.uint8)
        index = 0  # Index variable to count number of training binary files being processed.
        for file_name in files_names:

            if file_name[0:len(file_name) - 1] == "data_batch_":
                print("Working on : ", file_name)
                """
                """
                data_dict = unpickle_patch(dataset_path + file_name)
                """
                """
                images_data = data_dict[b"data"]
                # Reshaping all samples in the current binary file to be of 32x32x3 shape.
                images_data_reshaped = np.reshape(images_data,
                                                     newshape=(len(images_data), im_dim, im_dim, num_channels))
                # Appending the data of the current file after being reshaped.
                dataset_array[index * images_per_file:(index + 1) * images_per_file, :, :, :] = images_data_reshaped
                # Appening the labels of the current file.
                dataset_labels[index * images_per_file:(index + 1) * images_per_file] = data_dict[b"labels"]
                index = index + 1  # Incrementing the counter of the processed training files by 1 to accept new file.
        return dataset_array, dataset_labels  # Returning the training input data and output labels.


def unpickle_patch(file):
    """
    """
    patch_bin_file = open(file, 'rb')#Reading the binary file.
    patch_dict = pickle.load(patch_bin_file, encoding='bytes')#Loading the details of the binary file into a dictionary.
    return patch_dict#Returning the dictionary.


def create_model(session,forward_only):

    model=cnnModel.cnnModel(gConfig['percent'],gConfig['learning_rate'],gConfig['learning_rate_decay_factor'])
    if 'pretrained_model'in gConfig:
        model.saver.restore(session,gConfig['pretrained_model'])
        return model
    ckpt=tf.train.get_checkpoint_state(gConfig['working_directory'])

    if ckpt and ckpt.model_checkpoint_path:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

def get_batch(data,labels,percent):
    num_elements = np.uint32(percent * data.shape[0] / 100)
    shuffled_labels = labels
    np.random.shuffle(shuffled_labels)
    return data[shuffled_labels[:num_elements], :, :, :], shuffled_labels[:num_elements]


def train():
    # setup config to use BFC allocator
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    dataset_array, dataset_labels = read_data(dataset_path=gConfig['dataset_path'], im_dim=gConfig['im_dim'],
                                              num_channels=gConfig['num_channels'],num_files=gConfig['num_files'],images_per_file=gConfig['images_per_file'])
    print("Size of data : ", dataset_array.shape)
    with tf.Session(config=config) as sess:
        model=create_model(sess,False)
        # This is the training loop.
        step_time, correct = 0.0, 0.0
        current_step = 0
        previous_correct = []
        while True:
            start_time = time.time()
            shuffled_data, shuffled_labels = get_batch(data=dataset_array, labels=dataset_labels,
                                                             percent=30)
            step_correct=model.step(sess,shuffled_data,shuffled_labels,gConfig['keeps'],gConfig['dataset_size'],False)
            step_time += (time.time() - start_time) / gConfig['steps_per_checkpoint']
            correct += step_correct / gConfig['steps_per_checkpoint']
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % gConfig['steps_per_checkpoint'] == 0:
                #如果超过三次预测正确率没有升高则改变学习率
                if len(previous_correct) > 2 and correct < min(previous_correct[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_correct.append(correct)
                checkpoint_path = os.path.join(gConfig['working_directory'], "cnn.ckpt")
                model.saver.save(sess, checkpoint_path)
                print("在", str(gConfig['percent'] *gConfig['dataset_size']),"个样本集上训练的准确率", ' : ', correct)
                sys.stdout.flush()


def init_session(sess,conf='config.ini'):
    global gConfig
    gConfig=getConfig.get_config(conf)
    model=create_model(sess,True)
    return sess, model

def predict_line(sess,model,img):

    predict_name=model.step(sess,img,True)

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




