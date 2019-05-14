
# -*- coding: utf-8 -*-
import os
import sys
import string
import tempfile
import tensorflow as tf
import numpy as np
import getConfig
from tensorflow.keras.preprocessing import sequence
import textClassiferModel as model
import getConfig

gConfig={}

gConfig=getConfig.get_config(config_file='config.ini')

model_dir = gConfig['model_dir']
sentence_size=gConfig['sentence_size']
embedding_size = gConfig['embedding_size']
vocab_size=gConfig['vocabulary_size']

word_index=model.get_word_index(gConfig['vocabulary_file'])
word_inverted_index = {v: k for k, v in word_index.items()}

index_offset = 3
word_inverted_index[-1 - index_offset] = '_' # Padding at the end
word_inverted_index[ 1 - index_offset] = '>' # Start of the sentence
word_inverted_index[ 2 - index_offset] = '?' # OOV
word_inverted_index[ 3 - index_offset] = 'UNK'  # Un-used


def text_to_index(sentence):
    # Remove punctuation characters except for the apostrophe
    translator = str.maketrans('', '', string.punctuation.replace("'", ''))
    tokens = sentence.translate(translator).lower().split()
    print(tokens)
    return np.array([1] + [word_index[t] + index_offset if t in word_index else 2 for t in tokens])


def train():

	model.train_and_evaluate(model.cnn_pretrained_classifier)

def predict(sess,model,sentences):

    state=['pos','neg']

    indexes = [text_to_index(sentence) for sentence in sentences]
    x = sequence.pad_sequences(indexes, 
                               maxlen=gConfig['sentence_size'], 
                               padding='post', 
                               value=0)
   
    length = np.array([min(len(x), sentence_size) for x in indexes])
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x, "len": length}, shuffle=False)
    predictions = {}
    predict=[p['logistic'][0] for p in model.predict(input_fn=predict_input_fn)]
    if predict[0]>0.6:
       return state[0]
    else:
       return state[1]  

def init_session(sess,conf='config.ini'):
    global gConfig
    gConfig=getConfig.get_config(conf)
    cnn_pretrained_classifier = tf.estimator.Estimator(model_fn=model.cnn_model_fn,
                                        model_dir=os.path.join(model_dir, 'cnn_pretrained'),
                                        params=model.params)

    return sess,cnn_pretrained_classifier


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




