import os
from threading import Thread 
import time
import numpy as np
#from keras import Model
from tensorflow.keras import models
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
#from keras.models import model_from_json
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import messagebox 


def openNewWindow(filename):
    win2 = Toplevel(win1) 
    
    win2.title("Selected Image")  
    win2.geometry("400x400") 
    img = Image.open(filename['i'])
    img = img.resize((300,300))
    img = ImageTk.PhotoImage(img)
    panel = Label(win2, image = img).place(x=50,y=30)
    submit_button2 = Button(win2, text= "Run Classifier",command = lambda: run(filename)).place(x=150,y=350) 
    win2.mainloop()
        
def browseFiles(filename): 
    filename['i'] = filedialog.askopenfilename(initialdir = "/", 
                                          title = "Select Image", 
                                          filetypes = (("Image files", 
                                                        "*.jpg*"), 
                                                       ("all files", 
                                                        "*.*"))) 
    file_sel=filename['i']
    entryText.set(file_sel)
    
def process_image(filename):
    images=[] 
    img = cv2.imread(filename)
    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.
    images.append(img)
    images = np.stack(images, axis=0)
    return images
    
def run(filename):

    # load json and create model
    print("[INFO]       Loading model from disk")
    json_file = open('7cancermodel_Densenet201.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    newmodel = model_from_json(loaded_model_json)
    # load weights into new model
    newmodel.load_weights("7cancermodel_Densenet201.h5")
    print("[INFO]       Done")
    newmodel.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-5), metrics=['accuracy'])
    print("[INFO]       Compiled model")
    inimage=process_image(filename['i'])
    print("[INFO]       Classifying")
    predic= newmodel.predict(inimage)
    labels=['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis', 'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions']
    #print(predic.shape)
    print(labels)
    print(predic)
    classes = predic.argmax(axis=-1)
    n=classes[0]
    prob=int((predic[0,n])*100)
    #print(prob)

    if (prob)<75:
        messagebox.showinfo("info", "NOT SKIN CANCER")
    else:
        out=(labels[n]+"\n"+"\n"+"probability: "+str(prob)+"%")
        messagebox.showinfo("DETECTED", out)

 
 
def capture():
    
    camera = cv2.VideoCapture(0) #if there are two cameras, usually this is the front facing one
    if camera.read() == (False,None):
        camera= cv2.VideoCapture(0) 
    else:
        pass
    while True:
        return_value,image = camera.read()
        cv2.imshow('image',image)

        if cv2.waitKey(1)& 0xFF == ord('s'): #take a screenshot if 's' is pressed
            cv2.imwrite('capture'+'.jpg',image) #save screenshot as test.jpg
            break
        
        if cv2.waitKey(1)& 0xFF == ord('q'): #take a screenshot if 's' is pressedqq
            break

    camera.release()
    cv2.destroyAllWindows()

 
 
 
win1=Tk()
filename = {}
win1.geometry("520x100")
win1.title("Skin Cancer Classifier")
browse_button1 = Button(win1, text= "Browse", command = lambda: browseFiles(filename)).place(x=430,y=18)
entryText = tk.StringVar()
file_path_show = Entry(win1, width=40,font=12,textvariable = entryText ).place(x=20,y=20)
submit_button1 = Button(win1, text= "Submit",command = lambda: openNewWindow(filename) ).place(x=195,y=60)
capture_button2 = Button(win1, text= "Capture",command = lambda: capture() ).place(x=300,y=60)
#submit_button1 = Button(win1, text= "Submit",command = lambda: run(filename) ).place(x=195,y=60)
win1.mainloop()

