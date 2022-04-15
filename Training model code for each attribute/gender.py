from sklearn import metrics
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
img_array = cv2.imread(r"C:\\Users\Work\Desktop\\Age, Gender and Emotion recog to select relevant ads\\Datasets\\gender_dataset\\female\\131423.jpg.jpg")
print(img_array.shape)
Datadirectory= "C:\\Users\Work\Desktop\\Age, Gender and Emotion recog to select relevant ads\\Datasets\\Training\\"
Classes= ["female","male"]
for category in Classes:
    path= os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array= cv2.imread(os.path.join(path,img))
        plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break
img_size=224
new_array=cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array,cv2.COLOR_BGR2RGB))
plt.show()
training_Data=[]

def create_training_Data():
    for category in Classes:
        path=os.path.join(Datadirectory, category)
        class_num=Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                print("error")
                

create_training_Data()
print(len(training_Data))
temp= np.array(training_Data)
print(temp.shape)
import random
random.shuffle(training_Data)

X=[]
y=[]

for features,label in training_Data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,img_size,img_size,3)
X=X/255.0
Y=np.array(y)
import tensorflow as tf
from tensorflow import keras
from keras import layers

model = tf.keras.applications.MobileNetV2()
base_input= model.layers[0].input
base_output= model.layers[-2].output
base_output
final_output=layers.Dense(128)(base_output)
final_output=layers.Activation("relu")(final_output)
final_output=layers.Dense(64)(final_output)
final_output=layers.Activation("relu")(final_output)
final_output=layers.Dense(7,activation="softmax")(final_output)
final_output
new_model= keras.Model(inputs=base_input,outputs=final_output)
new_model.compile(loss="sparse_categorical_crossentropy",optimizer="adam", metrics= ["accuracy"])
history= new_model.fit(X,Y,epochs=10)
new_model.save("gender_model.h5")