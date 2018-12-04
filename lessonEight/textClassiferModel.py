# -*- coding: utf-8 -*-

"""
代码结构：

read_npz：读取npz文件
get_word_index：获取词典向量

train_input_fn：定义训练输入函数

eval_input_fn：定义测试输入函数

train_and_evaluate：训练主函数

cnn_model_fn：自定义CNNmodel函数

load_glove_embeddings：加载预训练好的参数

initializer：参数初始化函数

"""

import os
import string
import tempfile
import tensorflow as tf
import numpy as np
import getConfig
from tensorflow.keras.preprocessing import sequence
#建立词向量嵌入层，把输入文本转为可以进一步处理的数据格式

gConfig={}

gConfig=getConfig.get_config(config_file='config.ini')

tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.__version__)

sentence_size=gConfig['sentence_size']
embedding_size = gConfig['embedding_size']
vocab_size=gConfig['vocabulary_size']
model_dir = gConfig['model_dir']

def read_npz(data_file):
    r = np.load(data_file)
    return r['arr_0'],r['arr_1'],r['arr_2'],r['arr_3']

def get_word_index(vocabulary_file):

    tmp_vocab = []
    with open(vocabulary_file, "r") as f:#读取字典文件的数据，生成一个dict，也就是键值对的字典
         tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip() for line in tmp_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    return vocab
#下面这一段也可以用lessonTwo中的方式改写，大家可以尝试一下
word_index=get_word_index(gConfig['vocabulary_file'])
word_inverted_index = {v: k for k, v in word_index.items()}

index_offset = 3
word_inverted_index[-1 - index_offset] = '_' # Padding at the end
word_inverted_index[ 1 - index_offset] = '>' # Start of the sentence
word_inverted_index[ 2 - index_offset] = '?' # OOV
word_inverted_index[ 3 - index_offset] = 'UNK'  # Un-used

"""
sequence.pad_sequences(sequences,maxlen=None,dtype='int32',padding='pre',truncating='pre', value=0.)
sequences：浮点数或整数构成的两层嵌套列表

maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.在命名实体识别任务中，主要是指句子的最大长度

dtype：返回的numpy array的数据类型

padding：‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补

truncating：‘pre’或‘post’，确定当需要截断序列时，从起始还是结尾截断

value：浮点数，此值将在填充时代替默认的填充值0


"""


x_train_variable, y_train, x_test_variable, y_test =read_npz(gConfig['npz_data']) 

x_train = sequence.pad_sequences(x_train_variable, 
                                 maxlen=gConfig['sentence_size'], 
                                 padding='post', 
                                 value=0)
x_test = sequence.pad_sequences(x_test_variable, 
                                maxlen=gConfig['sentence_size'], 
                                padding='post', 
                                value=0)
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)


x_len_train = np.array([min(len(x), gConfig['sentence_size']) for x in x_train_variable])
x_len_test = np.array([min(len(x), gConfig['sentence_size']) for x in x_test_variable])

def parser(x, length, y):
    features = {"x": x, "len": length}
    return features, y
"""
tf.data.Dataset:
tf.data.Dataset.from_tensor_slices真正作用是切分传入Tensor的第一个维度，生成相应的dataset，即第一维表明数据集中数据的数量，之后切分batch等操作都以第一维为基础。
map和python中的map类似，map接收一个函数，Dataset中的每个元素都会被当作这个函数的输入，并将函数返回值作为新的Dataset，1981 3 173 4859 3 192 52 442 15 3 3 3664 7 10 1886 1937 125 271 314 16 54 3 3 3 154 15 3 3 877 302 188 3 5 3525 219 67 626 50 1 158 11 3 17 135 16 1 20 289 3 18 31 199 1 1432 7 161 1 22 9 11 13 25 1940 142 14 116 14 11 87 24 3 17 236 9 11 6 3 29 2 239 4137 4165 3 683 4137 3 57 3 2478 17 20 6 271 2 3 53 16 187 47 68 563 442 29 3 3 3 3
batch就是将多个元素组合成batch，如上所说，按照输入元素第一个维度，
repeat的功能就是将整个序列重复多次，主要用来处理机器学习中的epoch，假设原先的数据是一个epoch
"""

def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_train, x_len_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(x_train_variable))
    dataset = dataset.batch(100)
    dataset = dataset.map(parser)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_test, x_len_test, y_test))
    dataset = dataset.batch(100)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

column = tf.feature_column.categorical_column_with_identity('x', vocab_size)

all_classifiers = {}
def train_and_evaluate(classifier):
    # Save a reference to the classifier to run predictions later
    all_classifiers[classifier.model_dir] = classifier
    classifier.train(input_fn=train_input_fn, steps=25000)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    predictions = np.array([p['logistic'][0] for p in classifier.predict(input_fn=eval_input_fn)])   
    # Reset the graph to be able to reuse name scopes
    tf.reset_default_graph() 
    # Add a PR summary in addition to the summaries that the classifier writes
    pr = summary_lib.pr_curve('precision_recall', predictions=predictions, labels=y_test.astype(bool), num_thresholds=21)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(os.path.join(classifier.model_dir, 'eval'), sess.graph)
        writer.add_summary(sess.run(pr), global_step=0)
        writer.close()

word_embedding_column = tf.feature_column.embedding_column(column, dimension=embedding_size)

head = tf.contrib.estimator.binary_classification_head()

def cnn_model_fn(features, labels, mode, params):    
    input_layer = tf.contrib.layers.embed_sequence(
        features['x'], vocab_size, embedding_size,
        initializer=params['embedding_initializer'])
    
    training = mode == tf.estimator.ModeKeys.TRAIN
    dropout_emb = tf.layers.dropout(inputs=input_layer, 
                                    rate=0.2, 
                                    training=training)

    conv = tf.layers.conv1d(
        inputs=dropout_emb,
        filters=32,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)
    
    # Global Max Pooling
    pool = tf.reduce_max(input_tensor=conv, axis=1)
    
    hidden = tf.layers.dense(inputs=pool, units=250, activation=tf.nn.relu)
    
    dropout_hidden = tf.layers.dropout(inputs=hidden, 
                                       rate=0.2, 
                                       training=training)
    
    logits = tf.layers.dense(inputs=dropout_hidden, units=1)
    
    # This will be None when predicting
    if labels is not None:
        labels = tf.reshape(labels, [-1, 1])
        

    optimizer = tf.train.AdamOptimizer()
    
    def _train_op_fn(loss):
        return optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

    return head.create_estimator_spec(
        features=features,
        labels=labels,
        mode=mode,
        logits=logits, 
        train_op_fn=_train_op_fn)
  
params = {'embedding_initializer': tf.random_uniform_initializer(-1.0, 1.0)}
cnn_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                        model_dir=os.path.join(model_dir, 'cnn'),
                                       params=params)

def load_glove_embeddings(path):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            w = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embeddings[w] = vectors

    embedding_matrix = np.random.uniform(-1, 1, size=(vocab_size, embedding_size))
    num_loaded = 0
    for w, i in word_index.items():
        v = embeddings.get(w)
        if v is not None and i < vocab_size:
            embedding_matrix[i] = v
            num_loaded += 1
    embedding_matrix = embedding_matrix.astype(np.float32)
    return embedding_matrix

embedding_matrix = load_glove_embeddings('glove.6B.50d.txt')

"""To create a CNN classifier that leverages pretrained embeddings, we can reuse our `cnn_model_fn` but pass in a custom initializer that initializes the embeddings with our pretrained embedding matrix."""

def initializer(shape=None, dtype=tf.float32, partition_info=None):
    assert dtype is tf.float32
    return embedding_matrix

params = {'embedding_initializer': initializer}
cnn_pretrained_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn,
                                        model_dir=os.path.join(model_dir, 'cnn_pretrained'),
                                        params=params)


