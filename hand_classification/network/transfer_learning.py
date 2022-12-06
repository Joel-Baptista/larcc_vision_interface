import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from vision_config.vision_definitions import ROOT_DIR


def dataset():

    data = pd.read_csv(ROOT_DIR + f'/Datasets/MNIST/sign_mnist_train.csv')
    x_train = []
    y_train = []

    for i in range(0, len(data["label"])):
        sample_in_line = np.array(data.iloc[i][1:])

        sample_in_array = sample_in_line.reshape((28, 28))

        res = cv2.resize(sample_in_array.astype(np.uint8), (32, 32), interpolation=cv2.INTER_LINEAR)

        sample = np.zeros((32, 32, 3))
        sample[:, :, 0] = res
        sample[:, :, 1] = res
        sample[:, :, 2] = res

        label = [0] * 24

        if data.iloc[i][0] >= 10:
            index = data.iloc[i][0] - 1
        else:
            index = data.iloc[i][0]

        label[index] = 1

        x_train.append(sample)
        y_train.append(label)

    return x_train, y_train


if __name__ == '__main__':

    x_t, y_t = dataset()

    IMG_SHAPE = (32, 32, 3)

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(24)(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(x=np.array(x_t),
                        y=np.array(y_t),
                        epochs=50,
                        validation_split=0.2,
                        shuffle=True,
                        callbacks=[callback],
                        batch_size=2000,
                        verbose=2)

