

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_provider
import networks

tfgan = tf.contrib.gan
import getConfig
gConfig={}
gConfig=getConfig.get_config(config_file='config.ini')

"""
NHWC:
N代表数量， C代表channel，H代表高度，W代表宽度。

"""

def _define_model(images_x, images_y):
  """
这里我们定义一个cycleganmodel，需要用到高级API tfgan.cyclegan_model


tfgan.eval.add_cyclegan_image_summaries:


  """
  cyclegan_model = tfgan.cyclegan_model(
      generator_fn=networks.generator,
      discriminator_fn=networks.discriminator,
      data_x=images_x,
      data_y=images_y)

  # 为生成图片增加摘要.
  tfgan.eval.add_cyclegan_image_summaries(cyclegan_model)

  return cyclegan_model


"""

tf.train.polynomial_decay

polynomial_decay(
    learning_rate,
    global_step,
    decay_steps,
    end_learning_rate=0.0001,
    power=1.0,
    cycle=False,
    name=None
)

   earning_rate：初始值 
　　global_step：全局step数 
　　decay_steps：学习率衰减的步数,也代表学习率每次更新相隔的步数 
　　end_learning_rate：衰减最终值 
　　power：多项式衰减系数 
　　cycle：step超出decay_steps之后是否继续循环 

tf.cond:类似于ifelse语句
"""

def _get_lr(base_lr):
  
  global_step = tf.train.get_or_create_global_step()
  lr_constant_steps = gConfig['max_number_of_steps'] // 2

#这里定义学习率的自动下降功能，大家回忆一下我们前面几个课程是如何做的？

  def _lr_decay():
    return tf.train.polynomial_decay(
        learning_rate=base_lr,
        global_step=(global_step - lr_constant_steps),
        decay_steps=(gConfig['max_number_of_steps'] - lr_constant_steps),
        end_learning_rate=0.0)

  return tf.cond(global_step < lr_constant_steps, lambda: base_lr, _lr_decay)

"""
定义一个优化器，其参数只有一个lr，包括两个优化器，一个是生成器的，一个是识别器的。
tf.train.AdamOptimizer,这里的use_locking是True的说明是要加更新锁定
"""
def _get_optimizer(gen_lr, dis_lr):
 
  gen_opt = tf.train.AdamOptimizer(gen_lr, beta1=0.5, use_locking=True)
  dis_opt = tf.train.AdamOptimizer(dis_lr, beta1=0.5, use_locking=True)
  return gen_opt, dis_opt


def _define_train_ops(cyclegan_model, cyclegan_loss):
  """
  定以ops，Tensor，这个就是定义训练的运行计算图
  tf.contrib.gan.gan_train_ops


  tf.contrib.gan.gan_train_ops(
    model,
    loss,
    generator_optimizer,
    discriminator_optimizer,
    check_for_unused_update_ops=True,
    **kwargs
)

Args:
model: A GANModel.
loss: A GANLoss.
generator_optimizer: The optimizer for generator updates.
discriminator_optimizer: The optimizer for the discriminator updates.
check_for_unused_update_ops: If True, throws an exception if there are update ops outside of the generator or discriminator scopes.
**kwargs: Keyword args to pass directly to training.create_train_op for both the generator and discriminator train op.



  """
  gen_lr = _get_lr(gConfig['generator_lr'])
  dis_lr = _get_lr(gConfig['discriminator_lr'])
  gen_opt, dis_opt = _get_optimizer(gen_lr, dis_lr)
  train_ops = tfgan.gan_train_ops(
      cyclegan_model,
      cyclegan_loss,
      generator_optimizer=gen_opt,
      discriminator_optimizer=dis_opt,
      summarize_gradients=True,
      colocate_gradients_with_ops=True,
      aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

  tf.summary.scalar('generator_lr', gen_lr)
  tf.summary.scalar('discriminator_lr', dis_lr)
  return train_ops


"""
定义主函数

tf.device：

tf.train.replica_device_setter:


"""

def main(_):
  #safe编程，这里大家要注意一下，就是我们写的代码一定要避免出现这种OSbug，这里就是先判断文件夹是否存在，如果不存在就重现创建一下。
  if not tf.gfile.Exists(gConfig['train_log_dir']):
    tf.gfile.MakeDirs(gConfig['train_log_dir'])

  with tf.device(tf.train.replica_device_setter(gConfig['ps_tasks'])):
    with tf.name_scope('inputs'):
      images_x, images_y = data_provider.provide_custom_data(
          [gConfig['image_set_x_file_pattern'],gConfig['image_set_y_file_pattern']],
          batch_size=gConfig['batch_size'],
          patch_size=gConfig['patch_size'])
      # Set batch size for summaries.

      """
      set_shape:转变维度，将images_x转换成我们所需要的维度。
      tf.contrib.gan.cyclegan_loss：

      tf.contrib.gan.features.tensor_pool：

      tf.contrib.gan.GANTrainSteps:

      tf.train.StopAtStepHook:

      tf.train.LoggingTensorHook:

      tf.train.get_sequential_train_hooks:

      """
      images_x.set_shape([gConfig['batch_size'], None, None, None])
      images_y.set_shape([gConfig['batch_size'], None, None, None])

    # Define CycleGAN model.
    cyclegan_model = _define_model(images_x, images_y)
    
    # Define CycleGAN loss.
    cyclegan_loss = tfgan.cyclegan_loss(
        cyclegan_model,
        cycle_consistency_loss_weight=gConfig['cycle_consistency_loss_weight'],
        tensor_pool_fn=tfgan.features.tensor_pool)

    # Define CycleGAN train ops.
    train_ops = _define_train_ops(cyclegan_model, cyclegan_loss)

    # Training
    train_steps = tfgan.GANTrainSteps(1, 1)
    status_message = tf.string_join(
        [
            'Starting train step: ',
            tf.as_string(tf.train.get_or_create_global_step())
        ],
        name='status_message')
    if not gConfig['max_number_of_steps']:
      return
    tfgan.gan_train(
        train_ops,
        gConfig['train_log_dir'],
        get_hooks_fn=tfgan.get_sequential_train_hooks(train_steps),
        hooks=[
            tf.train.StopAtStepHook(num_steps=gConfig['max_number_of_steps']),
            tf.train.LoggingTensorHook([status_message], every_n_iter=10)
        ],
        master=gConfig['master'],
        is_chief=gConfig['task'] == 0)

if __name__ == '__main__':
  tf.app.run()
   