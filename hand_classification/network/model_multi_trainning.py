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



def create_model(model, include_top=False, classes=6, dropout=0.2):
    if model == "mobnetsmall":
        created_model = keras.applications.MobileNetV3Small(
            weights="imagenet",
            input_shape=(100, 100, 3),
            include_top=include_top
        )
    elif model == "mobnetlarge":
        created_model = keras.applications.MobileNetV3Large(
            input_shape=None,
            alpha=1.0,
            minimalistic=False,
            include_top=include_top,
            weights='imagenet',
            input_tensor=None,
            classes=classes,
            pooling=None,
            dropout_rate=dropout,
            classifier_activation='softmax',
            include_preprocessing=True
        )
    elif model == "mobnet":
        created_model = keras.applications.MobileNet(
            input_shape=None,
            alpha=1.0,
            depth_multiplier=1,
            dropout=dropout,
            include_top=include_top,
            weights="imagenet",
            input_tensor=None,
            pooling=None,
            classes=classes,
            classifier_activation="softmax",
            # **kwargs
        )
    elif model == "resnet":
        created_model = keras.applications.ResNet50(
            include_top=include_top,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=classes,
            # **kwargs
        )
    elif model == "vgg16":
        created_model = keras.applications.vgg16.VGG16(
            include_top=include_top,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=classes,
            classifier_activation="softmax",
        )
    elif model == "densenet":
        created_model = keras.applications.densenet.DenseNet121(
            include_top=include_top,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=classes,
            classifier_activation="softmax",
        )

    return created_model


if __name__ == '__main__':

    dataset_path = ROOT_DIR + "/Datasets/HANDS_dataset"

    x_data = []
    y_data = []

    gestures = ["G5", "G6"]
    # gestures = ["G1", "G2", "G3"]

    labels = ["G1_left", "G1_right", "G5_left", "G5_right", "G6_left", "G6_right"]

    l_r_index = 0
    count = 0
    print("Reading data")
    for subject in range(1, 6):
        print(f"Adding subject {subject}")
        for i, gesture in enumerate(gestures):

            data_path = f"/Subject{subject}/hand_segmented/{gesture}/output/"
            res = os.listdir(dataset_path + data_path)

            for file in res:

                if "left" in file:
                    l_r_index = 0
                elif "right" in file:
                    l_r_index = 1

                count += 1

                im = cv2.imread(dataset_path + data_path + file)

                im_array = np.asarray(im)

                if im_array.shape != (224, 224, 3):

                    print(im_array.shape)

                x_data.append(im_array)
                #
                # label = [0, 0, 0]
                # label[gesture - 1] = 1
                label = i * 2 + l_r_index
                # label = i
                y_data.append(label)

    print(f"There are {count} samples")
    print(f"x_data shape: {np.array(x_data).shape}")
    print(f"y_data shape: {np.array(y_data).shape}")
    print("Creating model")

    models = ["mobnetsmall",
              "mobnetlarge",
              "mobnet",
              "resnet",
              "vgg16",
              "densenet"]

    indexes = [3, 5]
    train_all_comb = [0, 10, 20, 30, 40, 50]

    i = 0

    for unfreeze_layers in train_all_comb:
        for index in indexes:

            base_model = create_model(models[index], classes=2 * len(gestures))

            # train_all = train_all_comb[i]
            # i += 1
            patience = 10

            callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

            num_layers = len(base_model.layers)

            for i, layer in enumerate(base_model.layers[:]):

                if i >= num_layers - unfreeze_layers:
                    layer.trainable = True
                else:
                    layer.trainable = False

            inputs = keras.Input(shape=(224, 224, 3))

            x = base_model(inputs, training=True)

            x = keras.layers.GlobalAveragePooling2D()(x)

            outputs = keras.layers.Dense(units=2 * len(gestures), activation='softmax')(x)

            model = keras.Model(inputs, outputs)

            model.compile(optimizer=keras.optimizers.Adam(),
                          loss=keras.losses.SparseCategoricalCrossentropy(),
                          metrics=['accuracy'])

            model_name = models[index] + f"_unfreeze{unfreeze_layers}"

            # print(f"Training was done with {sum(count)}\n{count[0]} gesture 1\n{count[1]} gesture 2\n{count[2]} gesture 3")
            model.summary()
            # input("Train model?")
            fit_history = model.fit(
                x=np.array(x_data),
                y=np.array(y_data),
                batch_size=200,
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

            # print(f"Training was done with {sum(count)}\n{count[0]} gesture 1\n{count[1]} gesture 2\n{count[2]} gesture 3")
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
