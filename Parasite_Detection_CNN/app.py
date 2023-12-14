#from flask import Flask, render_template, request
from fileinput import filename
from keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename



app = Flask(__name__)

cnn_model = load_model("models\\cnn_model.h5")
resenet_model=load_model("models\\resnet_model.h5")
vgg19_model=load_model("models\\vgg19_model.h5")


@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])

def get_output():

  if request.method == 'POST':

    img1 = request.files['my_image']

    imagefilename= img1.filename
    imgage_path = "static\\" + img1.filename

    print(imgage_path)	
    img1.save(imgage_path)
    img = cv2.imread(imgage_path)
    dim = (224, 224)

  # Resizing the image 
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA); 
    plt.grid(False) 


# Expanding the image dimensions 
    image = np.expand_dims(img, axis = 0); 


  # Making Final Predictions 
    result1 = cnn_model.predict(image)
    result2 = resenet_model.predict(image)
    result3 = vgg19_model.predict(image)
    print(result1,result2,result3)

    if result1 > 0.5:
      a='UnInfected'
    else:
      a='Infected'
      print(a)

    if result2 > 0.5:
      b='UnInfected'
    else:
      b='Infected'
      print(b)
    
    if result3 > 0.5:
      c='UnInfected'
    else:
      c='Infected'
      print(c)

    

    return render_template("index.html", prediction =(a, "Accuracy of CNN_MODEL: 54%"), prediction1=(b, "Accuracy of RESENET50: 94%"),prediction2=(c, "Accuracy of VGG19: 92%"),imagefilename=imagefilename,prediction_Imagename=imagefilename)




if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)