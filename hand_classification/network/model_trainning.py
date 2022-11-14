#!/usr/bin/env python3
import os
import cv2
from keras.applications.mobilenet_v3 import MobileNetV3Small, preprocess_input
from keras import Input, layers, optimizers, losses, metrics
import keras
from vision_config.vision_definitions import ROOT_DIR
import numpy as np
import matplotlib.pyplot as plt

dataset_path = ROOT_DIR + "/Datasets/HANDS_dataset"

x_data = []
y_data = []

print("Reading data")
for subject in range(1, 4):
    for gesture in range(1, 4):

        data_path = f"/Subject{subject}/Processed/G{gesture}/"
        res = os.listdir(dataset_path + data_path)

        for file in res:

            im = cv2.imread(dataset_path + data_path + file)

            im_array = np.asarray(im)

            x_data.append(im_array)

            label = [0, 0, 0]
            label[gesture-1] = 1

            y_data.append(label)


print("Creating model")
base_model = keras.applications.MobileNetV3Small(
    weights="imagenet",
    input_shape=(100, 100, 3),
    include_top=False
)

base_model.trainable = False

inputs = keras.Input(shape=(100, 100, 3))

x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)

outputs = keras.layers.Dense(3)(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


input("Train model?")
fit_history = model.fit(
    x=np.array(x_data),
    y=np.array(y_data),
    batch_size=50,
    epochs=10,
    verbose=2,
    callbacks=None,
    validation_split=0.2,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)


fig = plt.figure()

plt.subplot(1, 2, 1)
plt.plot(fit_history.history['accuracy'])
plt.plot(fit_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(fit_history.history['loss'])
plt.plot(fit_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.show()

model.save("myModel")
