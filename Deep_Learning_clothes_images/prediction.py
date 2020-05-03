# TensorFlow and Keras
import tensorflow.keras as tfk

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

fashion_mnist = tfk.datasets.fashion_mnist

#Download the images
(_, _), (test_images, _) = fashion_mnist.load_data()

test_images = test_images / 255.0

# se recupera el modelo
model = tfk.models.load_model('modelo_tf.h5')

#We make some predictions
predictions = model.predict(test_images)
# Show the prediction for the first image 
#(remeber the "class_names" array we defined before)
print(predictions[0])
