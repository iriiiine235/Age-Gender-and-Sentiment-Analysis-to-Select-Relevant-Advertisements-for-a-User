from tkinter import font
import tensorflow as tf
import cv2
import numpy as np
from cv2 import rectangle
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import webbrowser
emotion_model= tf.keras.models.load_model("emotion_model_mobilenetV2.h5")
gender_model= tf.keras.models.load_model("gender_model_mobilenetV2.h5")
age_model= tf.keras.models.load_model("age_model_mobilenetV2.h5")
import cv2
ad_dataset=r"C:\\Users\\Work\\.vscode\\Age-Gender-and-Sentiment-Analysis-to-Select-Relevant-Advertisements-for-a-User\\Advertisements.csv"
df = pd.read_csv(ad_dataset, encoding= 'unicode_escape')
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
        font=cv2.FONT_HERSHEY_PLAIN
        emotion_predictions= emotion_model.predict(final_image)
        gender_prediction= gender_model.predict(final_image)
        age_prediction= age_model.predict(final_image)
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
        emotion_status = emotions[np.argmax(emotion_predictions)]
        cv2.putText(frame, emotion_status, (x, y - 10), font, 0.7, (0, 255, 0), 2)
       
        # Display gender predictions on the frame
        gender_status = "Female" if np.argmax(gender_prediction) == 0 else "Male"
        cv2.putText(frame, gender_status, (x, y - 30), font, 0.7, (255, 0, 0), 2)

        # Display age predictions on the frame
        age_groups = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "70 or above"]
        age_status = age_groups[np.argmax(age_prediction)]
        cv2.putText(frame, age_status, (x, y - 50), font, 0.7, (0, 0, 255), 2)
       
        age_predict= np.argmax(age_prediction)
        final_age= str(age_predict)
        emotion_predict= np.argmax(emotion_predictions)
        final_emo= str(emotion_predict)
        gender_predict= np.argmax(gender_prediction)
        final_gender= str(gender_predict)
                             
       
   
    cv2.imshow("face emotion recognition", frame)
    if cv2.waitKey(2)& 0xFF==ord("q"):
        samples = df[df['AgeGroup'].str.contains(final_age)
                                    & df['Sentiment'].str.contains(final_emo)
                                    & df['Gender'].str.contains(final_gender)].sample(1)
                       
        link = list(samples['nd']).pop()
        if link:
                    print(f"Opening URL: {link}, predictions are saved @ frame.png")
                    webbrowser.open(link)
                    link = None
       
        break
   
cap.release()
cv2.destroyAllWindows()