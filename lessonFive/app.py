import pandas as pd
import numpy as np
import tensorflow as tf
import os
from flask import Flask, render_template, request, make_response
from flask import jsonify
from flask import Flask
from flask import jsonify
from flask import Response
import sys
import time  
import hashlib
import threading
import execute
import pandas as pd

app = Flask(__name__) 
#路由注解，我们这里使用的是path的形式进行传参
#
@app.route('/predict/<a>/<b>/<c>/<d>/<e>/<f>/<g>/<h>/<i>/<j>/<k>/<l>/<m>/<n>/<o>/<p>/<q>/<r>/<s>/<t>/<u>/<v>/<w>/<x>/<y>/<z>/<aa>/<ab>/<ac>/<ad>', methods=['GET'])
def predict(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac,ad):
	#获取url传来的需要进行预测的数据
	line=[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,ab,ac,ad]
	lines=range(31)
	lines=[int(i) for i in line]
	#因为我们全量的数据是31列，所以我们要在数据后面增加一个元素
	lines.append(0)
	COLUMNS = ['1','2','3', '4',  '5',  '6',  '7',  '8',  '9',  '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']

	#将数组转换成dataframe

	lines= pd.DataFrame([lines],columns=COLUMNS)
	
	#predict_result=train.predict(lines)
	predict_result=execute.predict(sess,lines,model)

	#返回
	return jsonify( { 'result of cluster': str(predict_result) } )

#初始化session，大家想一下如果不初始化会有什么问题？
sess = tf.Session()
sess, model = execute.init_session(sess, conf='config.ini')

if (__name__ == "__main__"): 
    app.run(host = '0.0.0.0', port = 8088) 


