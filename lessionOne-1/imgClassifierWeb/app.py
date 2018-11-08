"""
Good and quick.
Deeper in ML and DL.
Overfitting.
Github readme instructions should tell only how to run the code.
"""

import flask
import werkzeug
import os
import scipy.misc
import tensorflow as tf
import getConfig
import execute
gConfig={}
gConfig=getConfig.get_config(config_file='config.ini')

#Creating a new Flask Web application. It accepts the package name.
app = flask.Flask("imgClassifierWeb")

def CNN_predict():

    global sess
    global model
    global graph

    """
    global:
    """
    global secure_filename
    #从本地目录读取需要分类的图片
    img = scipy.misc.imread(os.path.join(app.root_path, secure_filename))

    """
    校验图片格式
    """
    if(img.ndim) == 3:
        """
        是否为32*32
        """
        if img.shape[0] == img.shape[1] and img.shape[0] == 32:
            """
            是否为3通道，GRB
            """
            if img.shape[-1] == 3:



                predicted_class = execute.predict_line(sess,model,img,graph)
                """
                将返回的结果用页面模板给渲染出来
                """
                return flask.render_template(template_name_or_list="prediction_result.html", predicted_class=predicted_class)
            else:
                """ 如果检测出图片格式不符合要求，则返回错误并返回上传图片的格式"""
                return flask.render_template(template_name_or_list="error.html", img_shape=img.shape)
        else:
            """ 如果检测出图片格式不符合要求，则返回错误并返回上传图片的格式"""
            return flask.render_template(template_name_or_list="error.html", img_shape=img.shape)
    return "遇到非图片格式的未知错误，请联系技术人员解决"
"""
app.add_url_rule:
"""
app.add_url_rule(rule="/predict/", endpoint="predict", view_func=CNN_predict)
"""
知识点：
flask.request.method
werkzeug

"""
def upload_image():
    global secure_filename
    if flask.request.method == "POST":#设置request的模式为POST
        img_file = flask.request.files["image_file"]#获取需要分类的图片
        secure_filename = werkzeug.secure_filename(img_file.filename)#生成一个没有乱码的文件名
        img_path = os.path.join(app.root_path, secure_filename)#获取图片的保存路径
        img_file.save(img_path)#将图片保存在应用的根目录下
        print("图片上传成功.")
        """
        
        """
        return flask.redirect(flask.url_for(endpoint="predict"))
    return "图片上传失败"

"""
"""
app.add_url_rule(rule="/upload/", endpoint="upload", view_func=upload_image, methods=["POST"])

def redirect_upload():

    return flask.render_template(template_name_or_list="upload_image.html")
"""
"""
app.add_url_rule(rule="/", endpoint="homepage", view_func=redirect_upload)
sess = tf.Session()
sess, model,graph = execute.init_session(sess, conf='config.ini')
if __name__ == "__main__":
    app.run(host="localhost", port=7777, debug=False)
