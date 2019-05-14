
import os
from absl import app
from absl import flags
import numpy as np
import PIL
import tensorflow as tf
from flask import Flask,render_template,request,redirect,url_for,make_response,jsonify
from werkzeug.utils import secure_filename
import cv2
from datetime import timedelta
import execute

from collections import Counter
import text2audio
import time

#设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
 
app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
 
@app.route('/maskImage', methods=['POST', 'GET'])  # 添加路由
def maskImage():
    if request.method == 'POST':
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
 
        user_input = request.form.get("name")
 
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
 
        upload_path = os.path.join(basepath, 'static/images',secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)
        trans_img_path,results=execute.maskImage(upload_path)
        r=results[0]
        class_ids=r['class_ids']
        labels=[]
        for i in range(len(class_ids)):
            label=execute.class_names[class_ids[i]]
            labels.append(label)
        #print(Counter(labels))
        labels=Counter(labels)
        strline=''.join(labels)
        print(strline)

        image_data = open(trans_img_path, "rb").read()
        response = make_response(image_data)
        response.headers['Content-Type'] = 'image/png'
        text2audio.read("在这张图片里有"+strline)
        return response
 
    return render_template('upload.html')

@app.route('/maskVideo', methods=['POST', 'GET']) 

def maskVideo():
    #execute.maskVideo()
    
      capture = cv2.VideoCapture(0)

    # these 2 lines can be removed if you dont have a 1080p camera.
      capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
      capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
      while True:
           ret, frame = capture.read()
           frame=execute.maskVideo(frame)
           cv2.imshow('frame', frame)
           #return make_response(frame)
      capture.release()
      cv2.destroyAllWindows()


from gevent.pywsgi import WSGIServer

http_server = WSGIServer(('',5000),app)
http_server.serve_forever()