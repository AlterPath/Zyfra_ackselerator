import glob
import random
import numpy as np
import pandas as pd
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

from tqdm import tqdm
from PIL import Image

from tensorflow.keras.utils import to_categorical

import seaborn as sns
import matplotlib.image as img
import matplotlib.pyplot as plt

train_df = pd.read_csv('Training_set.csv')
test_df = pd.read_csv('Testing_set.csv')
train_fol = glob.glob("./train/*")
test_fol = glob.glob("./test/*")
train_df.shape,test_df.shape
train_df.isna().sum(), test_df.isna().sum()
train_df.label.value_counts()
train_df.label.nunique()

import plotly.express as px
HAR = train_df.label.value_counts()
fig = px.pie(train_df, values=HAR.values, names=HAR.index,title="Label Distribution")
fig.show()
filename = train_df['filename']
labels = train_df['label']
def displaying_random():
    num = random.randint(1, 10000)
    image_filename =  f"Image_{num}.jpg"
    img_path = f"./train/{image_filename}"
    imgg = img.imread(img_path)
    plt.imshow(imgg)
    plt.title("{}".format(train_df.loc[train_df['filename'] == "{}".format(image_filename), 'label'].item()))
    plt.axis('off')
    plt.show()
displaying_random()
displaying_random()
displaying_random()
image_data = []
image_label = []

for i in (range(len(train_fol)-1)):
    t = './train/' + filename[i]
    imgg = Image.open(t)
    image_data.append(np.asarray(imgg.resize((160,160))))
    image_label.append(labels[i])
iii = image_data
iii = np.asarray(iii)
type(iii)
y_train = to_categorical(np.asarray(train_df["label"].factorize()[0]))
print(y_train[0])
efficientnet_model = Sequential()

model = tf.keras.applications.EfficientNetB7(include_top=False,
                                            input_shape=(160,160,3),
                                            pooling ="avg",classes=9,
                                             weights="imagenet")

for layer in model.layers:
    layer.trainable=False


efficientnet_model.add(model)
efficientnet_model.add(Flatten())
efficientnet_model.add(Dense(512,activation="relu"))
efficientnet_model.add(Dense(9,activation="softmax"))
efficientnet_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
history_efficientnet_model = efficientnet_model.fit(iii,y_train,epochs=40)
vgg_model = Sequential()

pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(160,160,3),
                   pooling='avg',classes=15,
                   weights='imagenet')

for layer in pretrained_model.layers:
        layer.trainable=False

vgg_model.add(pretrained_model)
vgg_model.add(Flatten())
vgg_model.add(Dense(256, activation='relu'))
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dense(15, activation='softmax'))
vgg_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
history_vgg16 = vgg_model.fit(iii,y_train, epochs=60)
