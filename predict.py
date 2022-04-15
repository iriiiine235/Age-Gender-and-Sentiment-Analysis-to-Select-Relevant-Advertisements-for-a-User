from tkinter import font
from grpc import Status
from sklearn import metrics
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from cv2 import rectangle
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import webbrowser
emotion_model= tf.keras.models.load_model("emotion_detection_model_100epochs.h5")
gender_model= tf.keras.models.load_model("gender_model.h5")
age_model= tf.keras.models.load_model("age_model.h5")
import cv2
path=r"C:\\Users\Work\Desktop\\Age, Gender and Emotion recog to select relevant ads\\haarcascade_frontalface_default.xml"
font_scale= 1.5
font=cv2.FONT_HERSHEY_PLAIN
rectangle_bgr=(255,255,255)
img=np.zeros((500,500))
text="Some text in a box"
(text_width,text_height)=cv2.getTextSize(text,font,fontScale=font_scale, thickness=1)[0]
text_offset_x=10
text_offset_y=img.shape[0]-25
box_coords= ((text_offset_x,text_offset_y),(text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x,text_offset_y), font,fontScale=font_scale, color=(0,0,0), thickness=1)
ad_dataset=r"C:\\Users\Work\Desktop\\Age, Gender and Emotion recog to select relevant ads\Advertisements.csv"
df = pd.read_csv(ad_dataset, encoding= 'unicode_escape')
cap= cv2.VideoCapture(1)
if not cap.isOpened():
    cap= cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces= faceCascade.detectMultiScale(gray, 1.1,4)
    for x,y,w,h in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color= frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y), (x+w,y+h),(255,0,0),2)
        faces= faceCascade.detectMultiScale(roi_gray)
        if len(faces)==0:
            print("Face not detected")
        else:
            for(ex,ey,ew,eh) in faces:
                face_roi= roi_color[ey:ey+eh,ex:ex+ew]
        final_image= cv2.resize(face_roi,(224,224))
        final_image= np.expand_dims(final_image,axis=0)
        final_image=final_image/255.0
        font=cv2.FONT_HERSHEY_SIMPLEX
        emotion_predictions= emotion_model.predict(final_image)
        font_scale=1.5
        font=cv2.FONT_HERSHEY_PLAIN
        gender_prediction= gender_model.predict(final_image)
        age_prediction= age_model.predict(final_image)
        disable=0

        if(np.argmax(emotion_predictions)==0):
            status="Angry"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
        elif (np.argmax(emotion_predictions)==1):
            status="Disgust"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
        elif (np.argmax(emotion_predictions)==2):
            status="Fear"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
        elif (np.argmax(emotion_predictions)==3):
            status="Happy"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
        elif (np.argmax(emotion_predictions)==4):
            status="Neutral"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))

        elif (np.argmax(emotion_predictions)==5):
            status="Sad"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
   
        else:
            status="Surprised"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,150),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
        if(np.argmax(age_prediction)==0):
            status="0-10"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,300),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
        elif (np.argmax(age_prediction)==1):
            status="11-20"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,300),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
        elif (np.argmax(age_prediction)==2):
            status="21-30"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,300),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
        elif (np.argmax(age_prediction)==3):
            status="31-40"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,300),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
        elif (np.argmax(age_prediction)==4):
            status="41-50"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,300),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))

        elif (np.argmax(age_prediction)==5):
            status="51-60"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,300),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
        elif (np.argmax(age_prediction)==6):
            status="61-70"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,300),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
   
        elif (np.argmax(age_prediction)==7):
            status="70 or above"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,300),font,3,(0,0,255),2,cv2.LINE_4)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
    
    
        if(np.argmax(gender_prediction)==0):
            status="female"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,200),font,3,(0,0,255),2,cv2.LINE_4)
        
    
        else:
            status="male"
            x1,y1,w1,h1= 0,0,175,75
            cv2.putText(frame,status,(100,200),font,3,(0,0,255),2,cv2.LINE_4)
        
        age_predict= np.argmax(age_prediction)
        final_age= str(age_predict)
        emotion_predict= np.argmax(emotion_predictions)
        final_emo= str(emotion_predict)
        gender_predict= np.argmax(gender_prediction)
        final_gender= str(gender_predict)
        DISABLE_DURATION=60
        DISABLE_DURATION = DISABLE_DURATION * cap.get(cv2.CAP_PROP_FPS)
                
         
                    
    
    cv2.imshow("face emotion recognition", frame)
    if cv2.waitKey(2)& 0xFF==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()


