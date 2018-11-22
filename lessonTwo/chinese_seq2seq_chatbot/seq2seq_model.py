0.# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

#前面进行了数据处理和准备工作，现在正式开始进行到核心部分，就是定义model，这个model是在tensorflow官方的model基础上进行了稍微的改动以便能够在
#tensorflow1.10上能够使用
#这里面我们主要会用到tensorflow和numpy以及我们之前定义的prepareData

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import是为了python尽量减小2.x和3.x之间的差异而存在的，简单来讲就是可以将下一个版本的特性加载到当前代码中使用

import random
import numpy as np
from six.moves import xrange  
import tensorflow as tf
import prepareData

#Seq2SeqModel从定义一个类开始，在这个类中我们会定义全部用的函数
#这个model也是整个工程的核心吧
#知识点：类、类的初始化


class Seq2SeqModel(object):
 
  def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm=False,
               num_samples=512, forward_only=False):
    """参数定义.

    超参数:
    在开始之前说一句，其实对于大部分的人工智能开发工程师或者研究者来讲更多的工作在超参数的调参上，因为
    这些超参决定着一个模型训练的结果和效果，所以我们花一点时间解释和讨论一下超参。
      source_vocab_size: 这个参数是决定我们采用的encoder数据字典的大小，这个参数以及下面那个target_vocab_size参数其实主要影响
      我们模型拟合的程度，什么意思那，其实就是表示我们在训练的时候认识多少字，如果我们在训练的时候对训练集中的字和词全部认识，那么拟合性会比较好，
      但是会造成过拟合，原因就是当我们在实际应用过程中如果遇到不认识的字和词就不能给出比较好的输出。但是如果太少的话，对于训练集来说，特征又太少所以会欠拟合.
      因此，大家在训练中可以根据实际情况来调整这些超参，以便让模型产生更加好的效果。
      target_vocab_size: 如上，已经解释。
      buckets:bukets可以理解成是是一个才有过滤器的容器，其目的就是把一定范围长度的输入、输出语句放在一起，
      这样做的原因是因为对于不同长度的输入输出是有不同的网络模型去训练，而我们实际生活中语句长度是各种各样的，如果每个不用bucket的话就会需要构建n多的神经网络，这样的模型是
      不可能收敛的，因此很自然的就用了长度分类的方法，将一定范围长度的数据放在一起，长短不齐的部分用PAD补全。
      对于一个bucket参数，因为我们的数据是有输入输出的，因为我们用(I,O)来定义，I表示的是在桶里的输入句子最长长度，O表示桶里输出的句子最长长度
      size: 这个超参数定义的是每层神经网络中的神经元的数量，神经元的数量和神经网络层数对整个模型的计算量右决定性的作用，一般来说神经元数量越多表达训练数据集的特征能力越强，拟合度越高.
      num_layers: 这个超参数定义了神经网络的层数.
      max_gradient_norm: 这个值非常关键，是用来解决梯度爆炸的，在梯度下降训练体系内有两个极端的问题需要解决的，一个是梯度爆炸另外一个就是梯度弥散（消失），这个值限制了梯度的最大值，防止梯度爆炸的发生.
      batch_size: 进行按批进行训练时，每批数据的大小，这里多说一下，因为我们训练的时候有非常大量的数据，我们不能一下子全部灌入计算图中进行计算因为这样既没有必要也不现实。
      learning_rate: 初始的学习率，简单解释一下学习率，学习率其实是一个系数，当学习率为1的时候，下个状态的值就等于当前状态加上梯度.换句话或学习率决定了模型的学习速率。
      learning_rate_decay_factor: 这个参数是设置学习率的衰减率,因为学习率越高训练速度越快但是拟合性不高，因此我们可以通过调节学习率的衰减来促进梯度下降.
      use_lstm: 对于神经单元是使用LSTM还是GRU，GRU是LSTM的变种版本，由于进行了门的合并，所以在计算效率上会增加，至于效果在不同的数据集上两者的表现各有千秋吧.
      num_samples: softmax是logistic基础上的推广，一般用在多标签分类中。但是如果标签过多，其实就涉及到计算量过大的问题，因此采用了采样的办法用词典中随机采样作为标注.
      forward_only: 这个是是否进行误差逆向传播计算的标志.
    """
    #在介绍完参数之后，我们需要对参数进行初始化，这里面有一个知识点需要讲一下，tf.Variable，tf.Varibale是用来创建变量的，在tf中的变量其实是可以通过运行操作来改变其值的张量。
    """
    知识点补充：
    tf.Variable参数：
    initial_value: A Tensor, or Python object convertible to a Tensor, which is the initial value for the Variable. The initial value must have a shape specified unless validate_shape is set to False. Can also be a callable with no argument that returns the initial value when called. In that case, dtype must be specified. (Note that initializer functions from init_ops.py must first be bound to a shape before being used here.)
    trainable: If True, the default, also adds the variable to the graph collection GraphKeys.TRAINABLE_VARIABLES. This collection is used as the default list of variables to use by the Optimizer classes.
    collections: List of graph collections keys. The new variable is added to these collections. Defaults to [GraphKeys.GLOBAL_VARIABLES].
    validate_shape: If False, allows the variable to be initialized with a value of unknown shape. If True, the default, the shape of initial_value must be known.
    caching_device: Optional device string describing where the Variable should be cached for reading. Defaults to the Variable's device. If not None, caches on another device. Typical use is to cache on the device where the Ops using the Variable reside, to deduplicate copying through Switch and other conditional statements.
    name: Optional name for the variable. Defaults to 'Variable' and gets uniquified automatically.
    variable_def: VariableDef protocol buffer. If not None, recreates the Variable object with its contents, referencing the variable's nodes in the graph, which must already exist. The graph is not changed. variable_def and the other arguments are mutually exclusive.
    dtype: If set, initial_value will be converted to the given type. If None, either the datatype will be kept (if initial_value is a Tensor), or convert_to_tensor will decide.
    expected_shape: A TensorShape. If set, initial_value is expected to have this shape.
    import_scope: Optional string. Name scope to add to the Variable. Only used when initializing from protocol buffer.
    constraint: An optional projection function to be applied to the variable after being updated by an Optimizer (e.g. used to implement norm constraints or value constraints for layer weights). The function must take as input the unprojected Tensor representing the value of the variable and return the Tensor for the projected value (which must have the same shape). Constraints are not safe to use when doing asynchronous distributed training.
    use_resource: if True, a ResourceVariable is created; otherwise an old-style ref-based variable is created. When eager execution is enabled a resource variable is always created.
    synchronization: Indicates when a distributed a variable will be aggregated. Accepted values are constants defined in the class tf.VariableSynchronization. By default the synchronization is set to AUTO and the current DistributionStrategy chooses when to synchronize. If synchronization is set to ON_READ, trainable must not be set to True.
    aggregation: Indicates how a distributed variable will be aggregated. Accepted values are constants defined in the class tf.VariableAggregation.

    """ 
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None

    """ 
    知识点：
    tf.get_variable（）

    tf.transpose（）数组转置，就是T。

   （a,b）元组，元组里的值是不能够改变的。

    tf.reshape(a,[m,n]),将a的维度变换成[m,n]，[-1,1]就是转换成一维数组。

    tf.nn.sampled_softmax_loss


    """
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w = tf.get_variable("proj_w", [size, self.target_vocab_size])
      w_t = tf.transpose(w)
      b = tf.get_variable("proj_b", [self.target_vocab_size])
      output_projection = (w, b)

      def sampled_loss(labels,logits):
        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(w_t, b, labels,logits,num_samples,self.target_vocab_size)
      
      softmax_loss_function = sampled_loss

    """
     知识点：
     这是我们在正式的系统课里会详细介绍的高阶API，这里我们只要知道其作用和如何使用就可以了。坦白说，这种高阶API对于我们开发者真的是福利，
     不需要理解苦涩难懂的数据推理公式和计算过程，以解决问题为导向，直接解决问题。
     tf.contrib.rnn.GRUCell
     tf.contrib.rnn.BasicLSTMCell
     tf.contrib.rnn.MultiRNNCell

    """

    # Create the internal multi-layer cell for our RNN.
    single_cell = tf.contrib.rnn.GRUCell(size)
    if use_lstm:
      single_cell = tf.contrib.rnn.BasicLSTMCell(size)
    cell = single_cell
    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)
    """

   知识点：
   embedding_attention
   1 2 3 4 6
   embedding俗称词嵌入，是一种进行特征降维的方式，主要是针对一些稀疏的数据。可以这样理解，比如 2 通过embedding可以得到一个向量 0.2,0.3,然后3  通过embedding可以得到一个向量 0.3,0.8，最后1 通过embedding可以得到一个向量 0.1,0.9，本来我表示1 2 3 需要
   三个特征维度，现在我用2个特征维度就可以表示，这样的数据就不会稀疏了，不稀疏才有可能进行数据拟合啊。     i
   attention也是一个非常有意思的东西，因为我们在自然语言处理中，词向量的顺序都是从左到右的，比如翻译 我 爱 包包 I Love BaoBao,其实我们可以看到 I 这个词与他前面的"包包"是没有很强的对应关系的
   反而I 与最前面的我是有非常对应的关系的，那么attention就是这样一个机制，就是可以计算每个对应的权重，然后让对应关系对应起来，这样的好处就是在进行BP的时候，可以直接从I 到 "包包"

   tf.contrib.legacy_seq2seq.embedding_attention_seq2seq

  encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
  decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
  cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
  num_encoder_symbols: Integer; number of symbols on the encoder side.
  num_decoder_symbols: Integer; number of symbols on the decoder side.
  embedding_size: Integer, the length of the embedding vector for each symbol.
  num_heads: Number of attention heads that read from attention_states.
  output_projection: None or a pair (W, B) of output projection weights and biases; W has shape [output_size x num_decoder_symbols] and B has shape [num_decoder_symbols]; if provided and feed_previous=True, each fed previous output will first be multiplied by W and added B.
  feed_previous: Boolean or scalar Boolean Tensor; if True, only the first of decoder_inputs will be used (the "GO" symbol), and all other decoder inputs will be taken from previous outputs (as in embedding_rnn_decoder). If False, decoder_inputs are used as given (the standard decoder case).
  dtype: The dtype of the initial RNN state (default: tf.float32).
  scope: VariableScope for the created subgraph; defaults to "embedding_attention_seq2seq".
  initial_state_attention: If False (default), initial attentions are zero. If True, initialize the attentions from the initial state and attention states.

  tf.placeholder：可以理解成形参，其作用是在定义流程时使用，等到具体计算的时候再赋值。

   """

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
          encoder_inputs, decoder_inputs, cell,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_size=size,
          output_projection=output_projection,
          feed_previous=do_decode)

    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]



    # Training outputs and losses.
    if forward_only:
        self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
        self.encoder_inputs, self.decoder_inputs, targets,
        self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
        softmax_loss_function=softmax_loss_function)
      # If we use output projection, we need to project outputs for decoding.
        if output_projection is not None:
           for b in xrange(len(buckets)):
             self.outputs[b] = [
              tf.matmul(output, output_projection[0]) + output_projection[1]
              for output in self.outputs[b]
          ]
    else:
      self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [prepareData.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([prepareData.GO_ID] + decoder_input +
                            [prepareData.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == prepareData.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights
