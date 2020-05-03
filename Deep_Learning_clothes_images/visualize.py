# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#Download again test_images
fashion_mnist = keras.datasets.fashion_mnist
(_, _), (test_images, _) = fashion_mnist.load_data()

#We visualize the first image as we predicted it before
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()