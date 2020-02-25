from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet, configure_uploads, ALL, DATA
import tensorflow as tf
from tensorflow.keras.metrics import top_k_categorical_accuracy 
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.backend import set_session
import cv2
import numpy as np
import os
from os.path import join, dirname, realpath 
import datetime
import time
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import cv2

app = Flask(__name__)
Bootstrap(app)

global sess
global graph

graph = tf.compat.v1.get_default_graph()

def top_2_accuracy(in_gt, in_pred): 
    return top_k_categorical_accuracy(in_gt, in_pred, k=2)

files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'uploads/'
configure_uploads(app,files)

def model_predict(img_path):
    
    img = image.load_img(img_path, target_size=(512, 512))  
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    with tf.compat.v1.Session(graph=graph) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        model = tf.keras.models.load_model("models/full_retina_model.h5", custom_objects={'top_2_accuracy': top_2_accuracy})
        preds = model.predict(x)
    
    #Convert list to str
    str1 = ""
    for i in range(0,5):
        str1 += str(round(preds[0][i],4))+","

    return str1


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/analyse', methods=['GET','POST'])
def analyse():

    if request.method == 'POST':
        # Get the file from post request
        fl = request.files['leftImageUpload']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        UPLOADS_PATH = join(dirname(realpath(__file__)), 'uploads/')
        
        if(fl.filename != ""):
            left_file_path = os.path.join(basepath, secure_filename(fl.filename))
            fl.save(left_file_path)
            resultleft = model_predict(left_file_path)
        return resultleft
    return None


if __name__ == '__main__':
	app.run()
	#http_server = WSGIServer(('0.0.0.0', 5000), app)
    #http_server.serve_forever()