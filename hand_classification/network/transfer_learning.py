#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from vision_config.vision_definitions import ROOT_DIR, exports, USERNAME
import copy
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import time
from transfer_learning_funcs import *


class TransferLearning:
    def __init__(self, config_path):
        self.dummy = None

        self.BATCH_SIZE = 100
        self.IMG_SIZE = (200, 200)

        self.base_model = "InceptionV3"
        self.pooling = "MaxPooling"
        self.model_name = f"{self.base_model}"
        self.training_epochs = 200
        self.training_batch_size = 2000
        self.training_patience = 10
        self.learning_rate = 0.0001
        self.val_split = 0.3

        with open(config_path) as json_file:
            self.config = json.load(json_file)

    def load_data(self, train_dir_path: str, test_dir_path: str):

        train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir_path,
                                                                    shuffle=True,
                                                                    batch_size=self.BATCH_SIZE,
                                                                    image_size=self.IMG_SIZE,
                                                                    label_mode='categorical',
                                                                    color_mode='rgb')

        test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir_path,
                                                                   shuffle=False,
                                                                   batch_size=self.BATCH_SIZE,
                                                                   image_size=self.IMG_SIZE,
                                                                   label_mode='categorical',
                                                                   color_mode='rgb')

        AUTOTUNE = tf.data.AUTOTUNE

        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

        print('Number of train batches: %d' % tf.data.experimental.cardinality(train_dataset))
        print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

        return train_dataset, test_dataset

    def get_features(self, feature_extractor, train_dataset, load_features=False):
        if not os.path.exists(f"/home/{USERNAME}/Datasets/extracted_features/{self.base_model}_{self.pooling}"):
            os.mkdir(f"/home/{USERNAME}/Datasets/extracted_features/{self.base_model}_{self.pooling}")

        extracted_features = None
        extracted_labels = None

        if load_features:
            extracted_features = np.load(f"/home/{USERNAME}/Datasets/extracted_features"
                                         f"/{self.base_model}_{self.pooling}/extracted_features.npy")
            extracted_labels = np.load(f"/home/{USERNAME}/Datasets/extracted_features"
                                       f"/{self.base_model}_{self.pooling}/extracted_labels.npy")

        if extracted_features is None or extracted_labels is None:

            extracted_features = None
            extracted_labels = None

            for i, batch in enumerate(iter(train_dataset)):

                image_batch = batch[0]
                label_batch = batch[1]

                feature_batch_average = feature_extractor(image_batch)
                print(i)
                if extracted_features is None:
                    extracted_features = feature_batch_average.numpy()
                    extracted_labels = label_batch.numpy()
                else:
                    extracted_features = np.concatenate((extracted_features, feature_batch_average.numpy()), axis=0)
                    extracted_labels = np.concatenate((extracted_labels, label_batch.numpy()), axis=0)

                print(extracted_features.shape)
                print(extracted_labels.shape)

            np.save(f"/home/{USERNAME}/Datasets/extracted_features/{self.base_model}_{self.pooling}"
                    f"/extracted_features.npy", extracted_features)
            np.save(f"/home/{USERNAME}/Datasets/extracted_features/{self.base_model}_{self.pooling}"
                    f"/extracted_labels.npy", extracted_labels)

        return tf.constant(extracted_features), tf.constant(extracted_labels)

    def train_model(self, model, data, labels):

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.training_patience)

        history = model.fit(x=data,
                            y=labels,
                            epochs=self.training_epochs,
                            validation_split=self.val_split,
                            shuffle=True,
                            verbose=1,
                            batch_size=self.training_batch_size,
                            callbacks=[callback])

        return model, history

    def create_model(self):
        img_shape = self.IMG_SIZE + (3,)

        if "ResNet50" in self.base_model:
            feature_extractor, features_input = create_resnet_base_model(img_shape, self.pooling)
        elif "MobileNetV2" in self.base_model:
            feature_extractor, features_input = create_mobilenetv2_base_model(img_shape, self.pooling)
        elif "InceptionV3" in self.base_model:
            feature_extractor, features_input = create_inceptionnet_base_model(img_shape, self.pooling)
        else:
            feature_extractor, features_input = create_mobilenetv2_base_model(img_shape, self.pooling)

        base_model = feature_extractor

        # Decision Model
        decision_input_shape = self.config[self.base_model][self.pooling]

        decision_inputs = tf.keras.Input(shape=(decision_input_shape,))
        x_d = tf.keras.layers.Dense(4)(decision_inputs)
        # x_d = tf.keras.activations.relu(x_d)
        # x_d = tf.keras.layers.Dropout(0.1)(x_d)
        # x_d = tf.keras.layers.Dense(128)(x_d)
        # x_d = tf.keras.activations.relu(x_d)
        # x_d = tf.keras.layers.Dense(4)(x_d)
        decision_outputs = tf.keras.activations.softmax(x_d)
        decision_model = tf.keras.Model(decision_inputs, decision_outputs)

        base_learning_rate = self.learning_rate

        decision_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                               loss=tf.keras.losses.CategoricalCrossentropy(),
                               metrics=['accuracy'])

        feature_extractor.summary()
        decision_model.summary()

        return base_model, decision_model

    def save_model(self, base_model, decision_model):

        outputs = decision_model(base_model.outputs)

        model = tf.keras.Model(base_model.inputs, outputs)
        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])

        model.save(ROOT_DIR + f"/hand_classification/models/{self.model_name}/myModel")

        # decision_model.save(ROOT_DIR + f"/hand_classification/network/{model_name}_decision/myModel")

        # training_stats = {"train_samples": extracted_features.shape[0],
        #                   "val_samples": int(extracted_features.shape[0] * self.val_split),
        #                   "test_samples": int(tf.data.experimental.cardinality(test_dataset) * BATCH_SIZE),
        #                   "feature_extraction_time": feature_time,
        #                   "training_time": training_time}
        #
        # print(training_stats)
        #
        # user_data_json = json.dumps(training_stats, indent=4)
        #
        # with open(ROOT_DIR + f"/hand_classification/network/{self.model_name}/training_data.json", "w") as outfile:
        #     outfile.write(user_data_json)

        return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Trains a new model based on a pre-trained model')
    parser.add_argument('-lf', '--load_features', action='store_true')
    args = vars(parser.parse_args())

    folder = "kinect/bg_removed"

    transfer_learning = TransferLearning(ROOT_DIR + '/hand_classification/config/transfer_learning.json')

    transfer_learning.base_model = "InceptionV3"
    transfer_learning.model_name = f"{transfer_learning.base_model}_bg_removed"

    model_freeze, model_train = transfer_learning.create_model()

    train_data, test_data = \
        transfer_learning.load_data(train_dir_path=f"/home/{USERNAME}/Datasets/ASL/bg_removed_augmented",
                                    test_dir_path=f"/home/{USERNAME}/Datasets/test_dataset/{folder}")

    features_train, labels_train = transfer_learning.get_features(model_freeze, train_data, args['load_features'])

    model_train, train_history = transfer_learning.train_model(model_train, features_train, labels_train)

    trained_model = transfer_learning.save_model(model_freeze, model_train)

    plot_curves(train_history)

    ground_truth, predictions, results = test_model(trained_model, test_data, ["A", "F", "L", "Y"],
                                                    folder, transfer_learning.model_name)

    plot_confusion(ground_truth, predictions, ["A", "F", "L", "Y"])

    # <=================================================================================================>
    # <====================================DATASET AUGMENTATION=========================================>
    # <=================================================================================================>

    # data_augmentation = tf.keras.Sequential([
    #     # tf.keras.layers.Resizing(224, 224, 'bilinear', False),
    #     # tf.keras.layers.RandomFlip('horizontal'),
    #     # tf.keras.layers.RandomRotation(0.1),
    #     # tf.keras.layers.RandomZoom((0.8, 1.5), None, "reflect", "bilinear")
    # ])

    # if not args["load_features"]:
    #     for image, _ in train_dataset.take(1):
    #         plt.figure(figsize=(10, 10))
    #         first_image = image[0]
    #         for i in range(9):
    #             ax = plt.subplot(3, 3, i + 1)
    #             augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    #             plt.imshow(augmented_image[0] / 255)
    #             plt.axis('off')

        # plt.show()

    # <=================================================================================================>
    # <===================================MODEL TESTING=================================================>
    # <=================================================================================================>

    # count_true = 0
    # count_false = 0
    # ground_truth = []
    # confusion_predictions = []
    # gestures = ["A", "F", "L", "Y"]
    # # gestures = ["G1", "G2", "G5", "G6"]
    #
    #
    # for i, batch in enumerate(iter(test_dataset)):
    #     # image_batch, label_batch = next(iter(test_dataset))
    #     image_batch = batch[0]
    #     label_batch = batch[1]
    #     feature_batch = feature_extractor(image_batch)
    #
    #     predictions = decision_model(feature_batch)
    #     print(i)
    #     for j, prediction in enumerate(predictions):
    #
    #         confusion_predictions.append(gestures[np.argmax(prediction)])
    #         ground_truth.append(gestures[np.argmax(label_batch[j])])
    #
    #         if np.argmax(prediction) == np.argmax(label_batch[j]):
    #             count_true += 1
    #         else:
    #             count_false += 1
    #
    # cm = confusion_matrix(ground_truth, confusion_predictions, labels=gestures)
    # blues = plt.cm.Blues
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gestures)
    # disp.plot(cmap=blues)
    #
    # print(f"Accuracy: {round(count_true / (count_false + count_true) * 100, 2)}%")
    # print(f"Tested with: {count_false + count_true}")
    #
    # plt.show()



