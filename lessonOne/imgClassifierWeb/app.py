import flask
import werkzeug
import os
import scipy.misc
import tensorflow as tf
import getConfig
import execute

gConfig = {}
gConfig = getConfig.get_config(config_file='config.ini')

# Creating a new Flask Web application. It accepts the package name.
app = flask.Flask("imgClassifierWeb")


def CNN_predict():
    global sess
    global model
    global graph

    """
    global:
    """
    global secure_filename
    # 从本地目录读取需要分类的图片
    img = scipy.misc.imread(os.path.join(app.root_path, secure_filename))

    """
    校验图片格式
    """
    if (img.ndim) == 3:
        """
        是否为32*32
        """
        if img.shape[0] == img.shape[1] and img.shape[0] == 32:
            """
            是否为3通道，GRB
            """
            if img.shape[-1] == 3:

                predicted_class = execute.predict_line(sess, model, img, graph)
                """
                将返回的结果用页面模板给渲染出来
                """
                return flask.render_template(template_name_or_list="prediction_result.html",
                                             predicted_class=predicted_class)
            else:
                """ 如果检测出图片格式不符合要求，则返回错误并返回上传图片的格式"""
                return flask.render_template(template_name_or_list="error.html", img_shape=img.shape)
        else:
            """ 如果检测出图片格式不符合要求，则返回错误并返回上传图片的格式"""
            return flask.render_template(template_name_or_list="error.html", img_shape=img.shape)
    return "遇到非图片格式的未知错误，请联系技术人员解决"


"""
flask路由系统：
1、使用flask.Flask.route() 修饰器。
2、使用flask.Flask.add_url_rule()函数。
3、直接访问基于werkzeug路由系统的flask.Flask.url_map.
参考知识链接：https://www.jianshu.com/p/e69016bd8f08
1、@app.route('/index.html')
    def index():
        return "Hello World!"
2、def index():
    return "Hello World!"
    index = app.route('/index.html')(index)
app.add_url_rule:app.add_url_rule(rule,endpoint,view_func)
关于rule、ednpoint、view_func以及函数注册路由的原理可以参考：https://www.cnblogs.com/eric-nirnava/p/endpoint.html
"""
app.add_url_rule(rule="/predict/", endpoint="predict", view_func=CNN_predict)
"""
知识点：
flask.request属性
form: 
一个从POST和PUT请求解析的 MultiDict（一键多值字典）。
args: 
MultiDict，要操作 URL （如 ?key=value ）中提交的参数可以使用 args 属性:
searchword = request.args.get('key', '')
values: 
CombinedMultiDict，内容是form和args。 
可以使用values替代form和args。
cookies: 
顾名思义，请求的cookies，类型是dict。
stream: 
在可知的mimetype下，如果进来的表单数据无法解码，会没有任何改动的保存到这个·stream·以供使用。很多时候，当请求的数据转换为string时，使用data是最好的方式。这个stream只返回数据一次。
headers: 
请求头，字典类型。
data: 
包含了请求的数据，并转换为字符串，除非是一个Flask无法处理的mimetype。
files: 
MultiDict，带有通过POST或PUT请求上传的文件。
method: 
请求方法，比如POST、GET
知识点参考链接：https://blog.csdn.net/yannanxiu/article/details/53116652
werkzeug
"""


def upload_image():
    global secure_filename
    if flask.request.method == "POST":  # 设置request的模式为POST
        img_file = flask.request.files["image_file"]  # 获取需要分类的图片
        secure_filename = werkzeug.secure_filename(img_file.filename)  # 生成一个没有乱码的文件名
        img_path = os.path.join(app.root_path, secure_filename)  # 获取图片的保存路径
        img_file.save(img_path)  # 将图片保存在应用的根目录下
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
sess, model, graph = execute.init_session(sess, conf='config.ini')
if __name__ == "__main__":
    app.run(host="localhost", port=7777, debug=False)
