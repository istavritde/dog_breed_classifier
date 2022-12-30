import json
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify,flash,redirect,url_for
import urllib.request
import os
import re
import string
from werkzeug.utils import secure_filename

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import cv2                

from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image                  
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import keras
import h5py
import tensorflow as tf

keras.backend.clear_session()
app = Flask(__name__)
app.secret_key= 'istavrit_dog_breed_prediction'
app.config["UPLOAD_FOLDER"] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH']= 16 * 1024 * 1024
ALLOWED_EXTENTIONS={'png','jpg','jpeg'}

global graph
graph = tf.get_default_graph() 

#initilize the model
# load face detector
face_cascade = cv2.CascadeClassifier("static/model_files/haarcascade_frontalface_alt.xml")
# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
model_file= "static/model_files/dog_breed_model.h5"
with h5py.File(model_file, 'a') as f:
    if 'optimizer_weights' in f.keys():
        del f['optimizer_weights']
        
InceptionV3_Model = keras.models.load_model("static/model_files/dog_breed_model.h5")
InceptionV3_Model.load_weights('static/model_files/weights.best.InceptionV3.hdf5')
### TODO: Compile the model.
InceptionV3_Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
dog_names = pd.read_csv("static/model_files/dog_names.csv",delimiter=",")['dog_names'].tolist()

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    with graph.as_default():
        prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENTIONS

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    return render_template('master.html')

def extract_InceptionV3(tensor):
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def InceptionV3_predict_breed(img_path):

    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    #predicted_vector = np.argmax(InceptionV3_Model.predict(np.expand_dims(bottleneck_feature, axis=0)))
    predicted_vector = InceptionV3_Model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    
    return dog_names[np.argmax(predicted_vector)]

def get_prediction(filename):

    if dog_detector(filename)==True:
        with graph.as_default():
            predicted_breed=InceptionV3_predict_breed(filename).split(".")[1]
    return "Predicted dog breed : {}".format(predicted_breed) 
    # if human is predicted, return resemblence
    if face_detector(filename)==True:
        predicted_breed=InceptionV3_predict_breed(filename).split(".")[1]
        return "This human looks like : {}".format(predicted_breed)
    #if neither is detected in the image, provide output that indicates an error.
    else:
        return "Could not detect any dog or human."

    
# web page that handles user query and displays model results
@app.route('/',methods=['POST'])
def upload_image():

    if request.method=='POST':

        if 'file' not in request.files:
            flash("Could not read file.")
            return redirect(request.url)

        file = request.files['file']

        if file.filename=='':
            flash("No image selected, please select an image.")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            flash("Image successfully uploaded")
            prediction = get_prediction(app.config['UPLOAD_FOLDER']+filename)
            return render_template("master.html",filename=filename,prediction=prediction)

        else:
            flash('Only image types of png,jpg and jpeg are allowed.')
            return redirect(request.url)

    else:
        redirect(request.url)

@app.route("/predict_display/<filename>")
def predict_display(filename):
    return redirect(url_for('static',filename='uploads/'+filename),code=301)
def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()