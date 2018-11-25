
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from absl import flags
import numpy as np
import PIL
import tensorflow as tf
import data_provider
import networks

import getConfig

from flask import Flask,render_template,request,redirect,url_for,make_response,jsonify
from werkzeug.utils import secure_filename
import os
import cv2
 
from datetime import timedelta


gConfig={}

gConfig=getConfig.get_config(config_file='config.ini')

tfgan = tf.contrib.gan


"""
这里定义一个判断函数，判断文件是否存在

"""
#文件操作，创建文件
def _make_dir_if_not_exists(dir_path):
  """Make a directory if it does not exist."""
  if not tf.gfile.Exists(dir_path):
    tf.gfile.MakeDirs(dir_path)

#文件操作，创建文件
def _file_output_path(dir_path, input_file_path):
  """Create output path for an individual file."""
  return os.path.join(dir_path, os.path.basename(input_file_path))

#定义一个预测图用来做预测
def make_inference_graph(model_name, patch_dim):
  """Build the inference graph for either the X2Y or Y2X GAN.

  Args:
    model_name: The var scope name 'ModelX2Y' or 'ModelY2X'.
    patch_dim: An integer size of patches to feed to the generator.

  Returns:
    Tuple of (input_placeholder, generated_tensor).
   知识点：
    tf.variable_scope：

    tf.get_variable(<name>, <shape>, <initializer>) 创建或返回给定名称的变量
    tf.variable_scope(<scope_name>) 管理传给get_variable()的变量名称的作用域

    tf.expand_dims：

    想要维度增加一维，可以使用tf.expand_dims(input, dim, name=None)函数。
    当然，我们常用tf.reshape(input, shape=[])也可以达到相同效果，
    但是有些时候在构建图的过程中，placeholder没有被feed具体的值，这时就会包下面的错误：TypeError: Expected binary or unicode string, got 1 
  """
  input_hwc_pl = tf.placeholder(tf.float32, [None, None, 3])

  # Expand HWC to NHWC
  images_x = tf.expand_dims(
      data_provider.full_image_to_patch(input_hwc_pl, patch_dim), 0)

  with tf.variable_scope(model_name):
    with tf.variable_scope('Generator'):
      generated = networks.generator(images_x)
  return input_hwc_pl, generated

#定义预测输出函数，将风格迁移后的图片输出到文件夹中

def export(sess, input_pl, output_tensor, input_file_pattern, output_dir):
  """Exports inference outputs to an output directory.

  Args:
    sess: tf.Session with variables already loaded.
    input_pl: tf.Placeholder for input (HWC format).
    output_tensor: Tensor for generated outut images.
    input_file_pattern: Glob file pattern for input images.
    output_dir: Output directory.
  """
  #如果output_dir配置了，要判断其是否存在
  if output_dir:
    _make_dir_if_not_exists(output_dir)


  """
tf.gfile.Glob

tf.gfile.Glob(filename)

查找匹配pattern的文件并以列表的形式返回，filename可以是一个具体的文件名，也可以是包含通配符的正则表达式。
关于gfile的操作可以查看：https://blog.csdn.net/a373595475/article/details/79693430


PIL:是一个python进行图片处理的库，可以参考文档http://effbot.org/imagingbook/

PIL.Image.fromarray

  """

  if input_file_pattern:
    for file_path in tf.gfile.Glob(input_file_pattern):
      # Grab a single image and run it through inference
      input_np = np.asarray(PIL.Image.open(file_path))
      output_np = sess.run(output_tensor, feed_dict={input_pl: input_np})
      image_np = data_provider.undo_normalize_image(output_np)
      output_path = _file_output_path(output_dir, file_path)
      PIL.Image.fromarray(image_np).save(output_path)
    return output_path


def trans(img_path):
   


  """
   这里提供了双方的风格迁移，一个是从一个风格迁移到我们想要的风格，另外一个是将我们迁移后的风格图片还原。
  """
  images_x_hwc_pl, generated_y = make_inference_graph('ModelX2Y',
                                                      gConfig['patch_dim'])
  images_y_hwc_pl, generated_x = make_inference_graph('ModelY2X',
                                                      gConfig['patch_dim'])

  # 重新加载我们在训练阶段保存的模型，然后进行图片的风格迁移
  saver = tf.train.Saver()
  with tf.Session() as sess:
    saver.restore(sess, gConfig['checkpoint_path'])
    #获取风格迁移周的图片路径
    img_path=export(sess, images_x_hwc_pl, generated_y,img_path ,
           gConfig['generated_y_dir'])
    export(sess, images_y_hwc_pl, generated_x, gConfig['image_set_y_glob'],
           gConfig['generated_x_dir'])

  print(img_path)
  return img_path

"""下面是一个APP应用，作用很简单就是将图片上传，并显示风格迁移后的图片"""


 
#设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
 
app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
 
@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
 
        user_input = request.form.get("name")
 
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
 
        upload_path = os.path.join(basepath, 'static/images',secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
        trans_img_path=trans(upload_path)

 
        image_data = open(trans_img_path, "rb").read()
        response = make_response(image_data)
        response.headers['Content-Type'] = 'image/png'
        return response
 
    return render_template('upload.html')



if __name__ == '__main__':
  #tf.app.run()
   app.run(host = '0.0.0.0',port = 8989,debug= False)

