from flask import Flask, render_template, url_for, request
import sqlite3
import os
import numpy as np
from tensorflow.keras.models import load_model #This function is crucial for loading pre-trained Keras models from disk.
import pickle #Python's Object Serialization Library
import shutil #Python's High-Level File Operations Module
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import  img_to_array , array_to_img
import numpy as np  # dealing with arrays

from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image





# Load the trained model
model = load_model('ResNet50_model.h5')

# Load class names from the pickle file
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

def predict_image(image):
    img =load_img(image, target_size=(150, 150))
    img_array =img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    print(predicted_class_index)
    predicted_class = class_names[predicted_class_index]
    print("predicted_class:",predicted_class)
    prediction1 = prediction.tolist()
    print(prediction1[0][predicted_class_index]*100)
    return predicted_class, prediction1[0][predicted_class_index]*100

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result)==0:
            return render_template('index.html',msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('home.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')



@app.route('/userlog.html', methods=['GET'])
def indexBt():
      return render_template('userlog.html')




@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/accuracy_plot.png',
              
              'http://127.0.0.1:5000/static/loss_plot.png']
    content=['Accuracy Graph',
            'Loss Graph']

            
    
        
    return render_template('graph.html',images=images,content=content)
    





@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        
        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        # #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        # #apply the Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        # #apply thresholding to segment the image (adaptive thresholding)
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        # # create the sharpening kernel( highpass filter)
        kernel_sharpening = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                    [-1,-1,-1]])

        # # apply the sharpening kernel to the image
        sharpened =cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg', sharpened)

        predicted_class, accuracy = predict_image("test/"+fileName)
        print("Predicted class:", predicted_class)
        print("Accuracy is:", accuracy)
       
        f = open('acc.txt', 'w')
        f.write(str(accuracy))
        f.close()

        
       
        str_label=""
        accuracy=""
        Tre=""
        Tre1=""
        if predicted_class =="MildDemented":
            str_label="MildDemented"
            Tre = "Medical Treatment"
            Tre1 = ["Cognitive Stimulation Activities",  
            "Healthy Diet", 
            "Regular Exercise", 
            "Medication and Treatment"]   
                    
            
        


        elif predicted_class =="ModerateDemented":
            str_label="ModerateDemented"
            Tre = "Medical Treatment"
            Tre1 =["Medications",  
            "Behavioral Interventions", 
            "Nutritional Support", 
            "Physical Exercise"]  


        elif predicted_class =="NonDemented":
            str_label="NonDemented"

        elif predicted_class =="VeryMildDemented":
            str_label="VeryMildDemented"
            Tre = "The remedies for  VeryMildDemented are:\n\n "
            Tre1 = ["Healthy Lifestyle Changes",  
            "Cognitive Stimulation", 
            "Medication Management", 
            "Behavioral Strategies"] 
            

       

       
        f = open('acc.txt', 'r')
        accuracy = f.read()
        f.close()
        print(accuracy)

        
        
        
        return render_template('results.html', status=str_label,status2=f'accuracy is {accuracy}',Treatment=Tre,Treatment1=Tre1,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg")
    return render_template('index.html')




@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
