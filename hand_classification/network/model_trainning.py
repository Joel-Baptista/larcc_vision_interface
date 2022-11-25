#!/usr/bin/env python3
import os
import cv2
from keras.applications.mobilenet_v3 import MobileNetV3Small, preprocess_input, MobileNetV3
from keras import Input, layers, optimizers, losses, metrics, callbacks
import keras
from keras.layers import Dense
from vision_config.vision_definitions import ROOT_DIR
import numpy as np
import matplotlib.pyplot as plt


def create_model(model):
    if model == "mobnetsmall":
        created_model = keras.applications.MobileNetV3Small(
            weights="imagenet",
            input_shape=(100, 100, 3),
            include_top=False
        )
    elif model == "mobnetlarge":
        created_model = keras.applications.MobileNetV3Large(
            input_shape=None,
            alpha=1.0,
            minimalistic=False,
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            classes=3,
            pooling=None,
            dropout_rate=0.2,
            classifier_activation='softmax',
            include_preprocessing=True
        )
    elif model == "mobnet":
        created_model = keras.applications.MobileNet(
            input_shape=None,
            alpha=1.0,
            depth_multiplier=1,
            dropout=0.001,
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            pooling=None,
            classes=3,
            classifier_activation="softmax",
            # **kwargs
        )
    elif model == "resnet":
        created_model = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=3,
            # **kwargs
        )
    elif model == "vgg16":
        created_model = keras.applications.vgg16.VGG16(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=3,
            classifier_activation="softmax",
        )

    return created_model


if __name__ == '__main__':

    dataset_path = ROOT_DIR + "/Datasets/HANDS_dataset"

    x_data = []
    y_data = []

    count = [0, 0, 0]
    print("Reading data")
    for subject in range(1, 6):
        print(f"Adding subject {subject}")
        for gesture in range(1, 4):

            data_path = f"/Subject{subject}/hand_segmented/G{gesture}/output/"
            res = os.listdir(dataset_path + data_path)

            for file in res:

                count[gesture - 1] += 1
                im = cv2.imread(dataset_path + data_path + file)

                im_array = np.asarray(im)

                x_data.append(im_array)
                #
                # label = [0, 0, 0]
                # label[gesture - 1] = 1
                label = gesture - 1
                y_data.append(label)


    print("Creating model")

    models = ["mobnetsmall",
              "mobnetlarge",
              "mobnet",
              "resnet",
              "vgg16"]

    indexes = [2]
    train_all_comb = [True]

    i = 0
    for index in indexes:

        base_model = create_model(models[index])

        model_name = models[index]
        train_all = train_all_comb[i]
        i += 1
        patience = 10

        if not train_all:
            model_name += "_frozen"

        callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

        base_model.trainable = train_all

        inputs = keras.Input(shape=(100, 100, 3))

        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)

        outputs = keras.layers.Dense(units=3, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

        print(f"Training was done with {sum(count)}\n{count[0]} gesture 1\n{count[1]} gesture 2\n{count[2]} gesture 3")
        model.summary()
        # input("Train model?")
        fit_history = model.fit(
            x=np.array(x_data),
            y=np.array(y_data),
            batch_size=2000,
            epochs=50,
            verbose=2,
            callbacks=[callback],
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

        print(f"Training was done with {sum(count)}\n{count[0]} gesture 1\n{count[1]} gesture 2\n{count[2]} gesture 3")
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

        model.save(ROOT_DIR + f"/hand_classification/network/{model_name}/myModel")

        plt.savefig(ROOT_DIR + f"/hand_classification/network/{model_name}/acc_plot.png")

        # plt.show()
