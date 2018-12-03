import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import getConfig
import sys

import os
gConfig={}
gConfig=getConfig.get_config(config_file='config.ini')
#定义训练数据的维度
seq_len=gConfig['seqlen']


#读取数据
def read_data(source_file):
    data=pd.read_csv(source_file,encoding='utf-8')
    dataset=data.values
    return dataset
dataset = read_data(gConfig['input_file'])

tf.reset_default_graph()

X_in = tf.placeholder(dtype=tf.float32, shape=[None, seq_len], name='X')
Y = tf.placeholder(dtype=tf.float32, shape=[None, seq_len], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, seq_len * 1])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
#encoder输出的数据维度
n_latent = gConfig['encoutlen']
#print(n_latent)
reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = 49 * dec_in_channels // 2

#定义激活函数

def lrelu(x, alpha=0.3):
   return tf.maximum(x, tf.multiply(x, alpha))
#定义编码机
def encoder(X_in, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
          X = tf.reshape(X_in, shape=[-1, seq_len, 1, 1])
          x1_layer = tf.layers.conv2d(X, filters=32, kernel_size=4, strides=2, padding='same', activation=activation)
          x1_dropout = tf.nn.dropout(x1_layer, keep_prob)
          x2_layer = tf.layers.conv2d(x1_dropout, filters=32, kernel_size=4, strides=2, padding='same', activation=activation)
          x2_dropout = tf.nn.dropout(x2_layer, keep_prob)
          x3_layer = tf.layers.conv2d(x2_dropout, filters=32, kernel_size=4, strides=1, padding='same', activation=activation)
          x3_dropout = tf.nn.dropout(x3_layer, keep_prob)
          x_out = tf.contrib.layers.flatten(x3_dropout)
          mn = tf.layers.dense(x_out, units=n_latent)
          sd = 0.5 * tf.layers.dense(x_out, units=n_latent)
          epsilon = tf.random_normal(tf.stack([tf.shape(x_out)[0], n_latent]))
          z  = mn + tf.multiply(epsilon, tf.exp(sd))
          return z, mn, sd

#定义解码机
def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
          y = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
          y = tf.layers.dense(y, units=inputs_decoder * 2 + 1, activation=lrelu)
          y = tf.reshape(y, reshaped_dim)
          y1_layer = tf.layers.conv2d_transpose(y, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
          y1_dropout = tf.nn.dropout(y1_layer, keep_prob)
          y2_layer = tf.layers.conv2d_transpose(y1_dropout, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
          y2_dropout = tf.nn.dropout(y2_layer, keep_prob)
          y3_layer = tf.layers.conv2d_transpose(y2_dropout, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)

          y_flatten = tf.contrib.layers.flatten(y3_layer)
          y_out = tf.layers.dense(y_flatten, units=seq_len * 1, activation=tf.nn.sigmoid)
          decoder_set = tf.reshape(y_out, shape=[-1, seq_len, 1])
          return decoder_set
#定义计算loss以及进行优化器优化的一系列tensor
global_step = tf.Variable(0, trainable=False)
sampled, mn, sd = encoder(X_in, keep_prob)
dec = decoder(sampled, keep_prob)
unreshaped = tf.reshape(dec, [-1, seq_len*1])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
dst_loss=img_loss+latent_loss
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(gConfig['learning_rate']).minimize(loss,global_step=global_step)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.all_variables())


def vae_train():       

#开始vae的训练
  for i in range(gConfig['vae_steps']):
     batch=dataset
#通过循环训练来通过优化器进行BP计算，进而更新参数使网络进行拟合，这里使用的是fullbatch模式
     sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
#根据checkpoint点对网络的收敛情况进行监测并保存下来模型     
     if not i % 200:
        ls, d, i_ls, d_ls, mu, sampled_data = sess.run([loss, dec, img_loss, dst_loss, mn, sampled], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
        print(i, ls, np.mean(i_ls), np.mean(d_ls))
        checkpoint_path = os.path.join(gConfig['working_directory'], "vae.ckpt")
        saver=tf.train.Saver()
        saver.save(sess,checkpoint_path,global_step=global_step)
#最后将训练集的所有的数据的encoder数据保存下来，用于接下来的kmeans训练
  sampled_data = sess.run(sampled, feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
  sampled_data=pd.DataFrame(sampled_data)
  sampled_data.to_csv(gConfig['sampled_path'])

#定义
def vae_encoder(sess,input_data):

  
  ckpt=tf.train.get_checkpoint_state(gConfig['working_directory'])
  if ckpt and ckpt.model_checkpoint_path:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()

  sampled_data = sess.run(sampled, feed_dict = {X_in: input_data, Y: input_data, keep_prob: 1.0})

  return sampled_data


if __name__=='__main__':
  if len(sys.argv) - 1:
     gConfig = getConfig(sys.argv[1])
  else:
        # get configuration from config.ini
     gConfig = getConfig.get_config()
  if gConfig['mode']=='train':
        vae_train()
  elif gConfig['mode']=='server':
     print('Sever Usage:python3 app.py')
      

