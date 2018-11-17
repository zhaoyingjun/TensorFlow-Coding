import tensorflow as tf
import numpy as np
import getConfig
import data_util
from functools import partial

import pandas as pd

gConfig={}

gConfig=getConfig.get_config(config_file='config.ini')

n_inputs =  1
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20  # codings
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs
learning_rate = 0.001

initializer = tf.contrib.layers.variance_scaling_initializer()
my_dense_layer = partial(
    tf.layers.dense,
    activation=tf.nn.elu,
    kernel_initializer=initializer)

X = tf.placeholder(tf.float32, [429398, n_inputs])
hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3_mean = my_dense_layer(hidden2, n_hidden3, activation=None)
hidden3_gamma = my_dense_layer(hidden2, n_hidden3, activation=None)
noise = tf.random_normal(tf.shape(hidden3_gamma), dtype=tf.float32)
hidden3 = hidden3_mean + tf.exp(0.5 * hidden3_gamma) * noise
hidden4 = my_dense_layer(hidden3, n_hidden4)
hidden5 = my_dense_layer(hidden4, n_hidden5)
logits = my_dense_layer(hidden5, n_outputs, activation=None)
outputs = tf.sigmoid(logits)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
reconstruction_loss = tf.reduce_sum(xentropy)
latent_loss = 0.5 * tf.reduce_sum(
    tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)
loss = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate=gConfig['learning_rate'])
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()



n_epochs = 50
batch_size = 150
def read_data(source_file):
    data=pd.read_csv(source_file,encoding='gbk')
    dataset=data.fillna(-1).values
    return dataset
def get_batch(data,percent):
  num_elements = np.uint32(percent * data.shape[0] / 100)
  shuffled_data = data
  np.random.shuffle(data)
  return data[shuffled_data[:num_elements]]

with tf.Session() as sess:

   init.run()
   k=gConfig['training_epochs']
   data_set=read_data(gConfig['input_file'])
   while k>0 :
      n_batches = int(100/gConfig['percent'])+1
      for iteration in range(n_batches):
        X_batch = data_set
        print (X_batch)
            #get_batch(data_set,gConfig['percent'])
        #sess.run(training_op, feed_dict={X: X_batch})
        loss_val, reconstruction_loss_val, latent_loss_val = sess.run([loss, reconstruction_loss, latent_loss], feed_dict={X: X_batch})
        print("\r{}".format(k), "Train total cost:", loss_val, "\tReconstruction loss:", reconstruction_loss_val, "\tLatent loss:", latent_loss_val)
        saver.save(sess, "./my_model_variational_variant.ckpt")
      k=k-1
   #codings = hidden3
  # saver.restore(sess,"./my_model_variational.ckpt")
  #codings_eval = codings.eval(feed_dict={X:data_set})

   


