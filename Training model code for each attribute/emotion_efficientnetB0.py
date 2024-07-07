from sklearn import metrics
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers
from keras import preprocessing
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalMaxPooling2D
from keras.models import Sequential
from keras.optimizers import gradient_descent_v2
from keras.preprocessing.image import ImageDataGenerator

Datadirectory= r"C:\\Users\\Work\\.vscode\\Age-Gender-and-Sentiment-Analysis-to-Select-Relevant-Advertisements-for-a-User\\Datasets\\emotion_recognition\\data\\train"
print(Datadirectory)
datagen= ImageDataGenerator(rescale=1./255, validation_split=0.2)
img_height=224
batch_size=32
train_ds=datagen.flow_from_directory(
 Datadirectory,
 subset="training",
 seed=123,
 class_mode="categorical",
 target_size=(img_height,img_height),
 batch_size=batch_size
)
val_ds =datagen.flow_from_directory(
  Datadirectory,
  subset="validation",
  seed=123,
  class_mode="categorical",
  target_size=(img_height,img_height),
  batch_size=batch_size

)


efficientnet_model = Sequential()

pretrained_model= tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224,224,3),
    pooling=None,
    classes=7
)

for layer in pretrained_model.layers:
    layer.trainable = False

efficientnet_model.add(pretrained_model)
efficientnet_model.add(GlobalMaxPooling2D())
efficientnet_model.add(Dense(512, activation="relu"))
efficientnet_model.add(Flatten())
efficientnet_model.add(Dense(7, activation="sigmoid"))
efficientnet_model.summary()
optimizer = gradient_descent_v2.SGD(learning_rate=0.001)
efficientnet_model.compile(optimizer= optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
epochs=20
history= efficientnet_model.fit(
    train_ds,
    validation_data= val_ds,
    epochs=epochs
)

efficientnet_model.save('emotion_model_efficientnetB0.h5')
flg1 = plt.gcf()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.axis(ymin=0.4,ymax=1)
plt.grid()
plt.title("EfficientNetB0 Model Accuracy - Emotion")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["train","validation"])
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.grid()
plt.title("EfficientNetB0 Model Loss - Emotion")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["train","validation"])
plt.show()

from keras.models import load_model


#Test the model
my_model = load_model('emotion_model_efficientnetB0.h5', compile=False)

#Generate a batch of images
test_img, test_lbl = val_ds.__next__()
predictions=my_model.predict(test_img)

predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_lbl, axis=1)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, predictions))

#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, predictions)
#print(cm)
import seaborn as sns
sns.heatmap(cm, annot=True)

import random

class_labels=['angry','disgust', 'fear', 'happy','neutral','sad','surprise']
#Check results on a few select images
n=random.randint(0, test_img.shape[0] - 1)
image = test_img[n]
orig_labl = class_labels[test_labels[n]]
pred_labl = class_labels[predictions[n]]
plt.imshow(image[:,:,0], cmap='gray')
plt.title("Original label is:"+orig_labl+" Predicted is: "+ pred_labl)
plt.show()