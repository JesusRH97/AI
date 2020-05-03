# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'

fashion_mnist = keras.datasets.fashion_mnist

#Download the images
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Create the classes 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#First of all, we have to resize the images. CNN must work
#with images between 0 and 1 values
train_images = train_images / 255.0
test_images = test_images / 255.0


#We define the model. In consists on three layers:
#The first one is the input layer and it flattens the image.
#The second and third one are the fully-connected layers with 128 neurons
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)])

#We have to compile it
model.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

#Itś time to train the model with the train_images dataset
model.fit(train_images, train_labels, epochs=10)


#We evaluate the accuracy of the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


#Finally, we save the model
model.save('modelo_tf.h5')

"""
#We make some predictions
predictions = model.predict(test_images)
# Show the prediction for the first image
predictions[0]
"""