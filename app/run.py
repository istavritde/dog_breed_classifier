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
import matplotlib.pyplot as plt  
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50

app = Flask(__name__)
app.secret_key= 'istavrit_dog_breed_prediction'
app.config["UPLOAD_FOLDER"] = 'app/static/uploads/'
app.config['MAX_CONTENT_LENGTH']= 16 * 1024 * 1024
ALLOWED_EXTENTIONS={'png','jpg','jpeg'}

#initilize the model
# load face detector
face_cascade = cv2.CascadeClassifier("app/static/model_files/haarcascade_frontalface_alt.xml")
# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

### Obtain bottleneck features from another pre-trained CNN.
bottleneck_features = np.load("app/static/model_files/DogInceptionV3Data.npz")
train_InceptionV3 = bottleneck_features['train']
valid_InceptionV3 = bottleneck_features['valid']
test_InceptionV3  = bottleneck_features['test']

### Define your architecture.

InceptionV3_Model = Sequential()

InceptionV3_Model.add(GlobalAveragePooling2D(input_shape=train_InceptionV3.shape[1:]))

InceptionV3_Model.add(Dense(133,activation='softmax'))

#InceptionV3_Model.summary()

### Compile the model.
InceptionV3_Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

### Load the model weights with the best validation loss.
InceptionV3_Model.load_weights('app/static/model_files/weights.best.InceptionV3.hdf5')
dog_names = pd.read_csv("app/static/model_files/dog_names.csv",delimiter=",")['dog_names'].tolist()

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENTIONS

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    return render_template('master.html')

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
            prediction = get_prediction(filename)
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