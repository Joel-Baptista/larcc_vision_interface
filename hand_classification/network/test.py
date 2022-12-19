import tensorflow as tf
from transfer_learning_funcs import *

img_shape = (200, 200, 3)
freeze_percent = 1

preprocess_input = tf.keras.applications.inception_v3.preprocess_input

base_model = tf.keras.applications.InceptionV3(input_shape=img_shape,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False
base_model.summary()

model1, model2 = split_keras_model(base_model, 0.9)

model1.summary()
model2.summary()
