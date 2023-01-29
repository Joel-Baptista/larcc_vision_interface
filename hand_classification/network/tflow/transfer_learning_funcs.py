#!/usr/bin/env python3
from vision_config.vision_definitions import ROOT_DIR, USERNAME
import os
import json
import imgaug.augmenters as iaa
import imgaug as ia
import numpy as np
import cv2
import os

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, \
    precision_score, f1_score
import pandas as pd
from vision_config.vision_definitions import ROOT_DIR, exports, USERNAME
import json


def plot_confusion(ground_truth, predictions, labels):
    """
    Plots the confusion matrix of test results
    Input:
      ground_truth: List of the labeled ground truth.
      predictions: List of the labeled model predictions
      lables: List containing all labeled categories
    Output:
      None
    """

    cm = confusion_matrix(ground_truth, predictions, labels=labels, normalize='true')
    cm = 100 * cm
    blues = plt.cm.Blues
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=blues)

    plt.show()


def test_model(model, test_dataset, labels, test_folder, model_name, dataset):
    """
    Test the model using test images not used in the train
    Input:
      model: Tensorflow trained model
      test_dataset: tf.keras.utils.image_dataset_from_directory object containing the test images
      labels: List containing all labeled categories
      test_folder: String containing the folder where the test data will be saved
      model_name: String containing the name of the model tested
    Output:
      ground_truth: List containing the ground truth labels of the tested images
      predcitions: List containing the predicted labels of the tested images
      dic_results: Dictionary containing the evaluation metrics (accuracy, precision, recall, f1_score) of the test
    """

    count_true = 0
    count_false = 0
    ground_truth = []
    confusion_predictions = []

    dic_test = {"index": [],
                "gesture": [],
                "prediction": [],
                }

    if not os.path.exists(f"/home/{USERNAME}/Datasets/{dataset}/{test_folder}/wrong"):
        os.mkdir(f"/home/{USERNAME}/Datasets/{dataset}/{test_folder}/wrong")
        for label in labels:
            os.mkdir(f"/home/{USERNAME}/Datasets/{dataset}/{test_folder}/wrong/{label}")

    for i, batch in enumerate(iter(test_dataset)):

        image_batch = batch[0]
        label_batch = batch[1]
        #
        # buffer = []
        #
        # for image in image_batch:
        #     # print(np.array(image)[1, 1:5, :].astype(np.uint8))
        #     cv2.imshow("Before", np.array(image).astype(np.uint8))
        #
        #     image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        #     # image = np.array(image).astype(np.uint8)
        #     # print(image[1, 1:5, :].astype(np.uint8))
        #     cv2.imshow("After", np.array(image).astype(np.uint8))
        #     cv2.waitKey(1)
        #     buffer.append(image)

        # predictions = model.predict(x=np.array(buffer), verbose=2)
        predictions = model.predict(x=np.array(image_batch), verbose=2)
        # print(label_batch)
        for j, prediction in enumerate(predictions):

            # print(prediction)
            index = j + len(image_batch) * i

            dic_test["index"].append(index)
            dic_test["prediction"].append(labels[np.argmax(prediction)])
            dic_test["gesture"].append(labels[np.argmax(label_batch[j])])

            confusion_predictions.append(labels[np.argmax(prediction)])
            ground_truth.append(labels[np.argmax(label_batch[j])])

            if np.argmax(prediction) == np.argmax(label_batch[j]):
                count_true += 1
            else:
                count_false += 1
                cv2.imwrite(f"/home/{USERNAME}/Datasets/{dataset}/"
                            f"{test_folder}/wrong/{labels[np.argmax(prediction)]}"
                            f"/image{index}.png", cv2.cvtColor(np.array(image_batch[j]), cv2.COLOR_RGB2BGR))

    df = pd.DataFrame(dic_test)
    df.to_csv(f"/home/{USERNAME}/Datasets/{dataset}/{test_folder}/{model_name}_test_results.csv")

    print(f"Accuracy: {round(count_true / (count_false + count_true) * 100, 2)}%")
    print(f"Tested with: {count_false + count_true}")

    recall = recall_score(ground_truth, confusion_predictions, average=None)
    precision = precision_score(ground_truth, confusion_predictions, average=None)
    f1 = f1_score(ground_truth, confusion_predictions, average=None)

    dic_results = {"accuracy": accuracy_score(ground_truth, confusion_predictions), "precision": {},
                   "recall": {}, "f1": {}}

    for i, label in enumerate(labels):
        print(i)
        dic_results["recall"][label] = recall[i]
        dic_results["precision"][label] = precision[i]
        dic_results["f1"][label] = f1[i]

    dic_results["recall"]["average"] = np.mean(recall)
    dic_results["precision"]["average"] = np.mean(precision)
    dic_results["f1"]["average"] = np.mean(f1)

    print(accuracy_score(ground_truth, confusion_predictions))
    print(recall_score(ground_truth, confusion_predictions, average=None))
    print(precision_score(ground_truth, confusion_predictions, average=None))
    print(f1_score(ground_truth, confusion_predictions, average=None))

    print(dic_results)

    with open(f"/home/{USERNAME}/Datasets/{dataset}/{test_folder}/{model_name}_results.json", 'w') as outfile:
        json.dump(dic_results, outfile)

    return ground_truth, confusion_predictions, dic_results


def plot_curves(history):
    """
    Plot the validation accuracy and loss of the train
    Input:
        history: TensorFlow object returned by the model.fit function
    Output:
        None
    """

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 8.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')

    plt.show()


def get_bottom_top_model(model, layer_name):
    layer = model.get_layer(layer_name)
    bottom_input = tf.keras.Input(model.input_shape[1:])
    bottom_output = bottom_input
    top_input = tf.keras.Input(layer.output_shape[1:])
    top_output = top_input

    bottom = True
    for layer in model.layers:
        if bottom:
            bottom_output = layer(bottom_output)
        else:
            top_output = layer(top_output)
        if layer.name == layer_name:
            bottom = False

    bottom_model = tf.keras.Model(bottom_input, bottom_output)
    top_model = tf.keras.Model(top_input, top_output)

    return bottom_model, top_model


def split_keras_model(model, freeze_percent):
    """
    Input:
      model: A pre-trained Keras Sequential model
      index: The index of the layer where we want to split the model
    Output:
      model1: From layer 0 to index
      model2: From index+1 layer to the output of the original model
    The index layer will be the last layer of the model_1 and the same shape of that layer will be the input layer of the model_2
    """

    index = int(freeze_percent * len(model.layers))
    # Creating the first part...
    # Get the input layer shape
    layer_input_1 = tf.keras.Input(model.layers[0].input_shape[1:])
    # Initialize the model with the input layer
    x = layer_input_1
    # Foreach layer: connect it to the new model
    for layer in model.layers[1:index]:
        x = layer(x)
    # Create the model instance
    model1 = tf.keras.Model(inputs=layer_input_1, outputs=x)

    # Creating the second part...
    # Get the input shape of desired layer
    input_shape_2 = model.layers[index].get_input_shape_at(0)[1:]
    print("Input shape of model 2: " + str(input_shape_2))
    # A new input tensor to be able to feed the desired layer
    layer_input_2 = tf.keras.Input(shape=input_shape_2)

    # Create the new nodes for each layer in the path
    x = layer_input_2
    # Foreach layer connect it to the new model
    for layer in model.layers[index:]:
        x = layer(x)

    # create the model
    model2 = tf.keras.Model(inputs=layer_input_2, outputs=x)

    return model1, model2


def create_extraction_layer(extraction_type: str) -> tf.keras.layers:

    if extraction_type == "Flatten":
        return tf.keras.layers.Flatten()
    elif extraction_type == "MaxPooling":
        return tf.keras.layers.GlobalAveragePooling2D()

    return tf.keras.layers.GlobalAveragePooling2D()


def create_inceptionnet_base_model(img_shape: tuple, extraction_type: str):
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input

    base_model = tf.keras.applications.InceptionV3(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False

    extraction_layer = create_extraction_layer(extraction_type)

    feature_inputs = tf.keras.Input(shape=img_shape)
    x_f = preprocess_input(feature_inputs)
    x_f = base_model(x_f, training=False)
    features = extraction_layer(x_f)
    extractor = tf.keras.Model(feature_inputs, features)

    return extractor, feature_inputs


def create_resnet_base_model(img_shape: tuple, extraction_type: str):

    preprocess_input = tf.keras.applications.resnet50.preprocess_input

    base_model = tf.keras.applications.ResNet50(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')

    base_model.trainable = False

    extraction_layer = create_extraction_layer(extraction_type)

    feature_inputs = tf.keras.Input(shape=img_shape)
    x_f = preprocess_input(feature_inputs)
    x_f = base_model(x_f, training=False)
    features = extraction_layer(x_f)
    extractor = tf.keras.Model(feature_inputs, features)

    return extractor, feature_inputs


def create_mobilenetv2_base_model(img_shape: tuple, extraction_type: str):
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    extraction_layer = create_extraction_layer(extraction_type)

    feature_inputs = tf.keras.Input(shape=img_shape)
    x_f = preprocess_input(feature_inputs)
    x_f = base_model(x_f, training=False)
    features = extraction_layer(x_f)
    extractor = tf.keras.Model(feature_inputs, features)

    return extractor, feature_inputs

