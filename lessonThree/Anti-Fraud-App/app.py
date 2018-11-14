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

#
app = Flask(__name__) 
#路由注解，我们这里使用的是path的形式进行传参
#示例url:http://0.0.0.0:8088/predict/1/2/3/4/5/6/7/8/9/10/11/12 这里的1...12换成需要进行聚类的值就可以了
@app.route('/predict/<a>/<b>/<c>/<d>/<e>/<f>/<g>/<h>/<i>/<j>/<k>/<l>', methods=['GET'])
def predict(a,b,c,d,e,f,g,h,i,j,k,l):
	line=[a,b,c,d,e,f,g,h,i,j,k,l]
	lines=range(12)
	lines=[int(i) for i in line]
	lines=[lines]
	predict_result=execute.predicts(lines)

	return jsonify( { 'result of cluster': str(predict_result) } )

if (__name__ == "__main__"): 
    app.run(host = '0.0.0.0', port = 8088) 