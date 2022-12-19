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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Trains a new model based on a pre-trained model')
    parser.add_argument('-lf', '--load_features', action='store_true')
    args = vars(parser.parse_args())
    print(args)

    with open(ROOT_DIR + '/hand_classification/config/transfer_learning.json') as json_file:
        config = json.load(json_file)

    #<=================================================================================================>
    #<====================================DATASET PREPARATION==========================================>
    #<=================================================================================================>

    exports() # Function for CUDA cores

    BATCH_SIZE = 1000
    IMG_SIZE = (200, 200)

    PATH = f"/home/{USERNAME}/Datasets/ASL"

    # train_dir = os.path.join(PATH, 'asl_alphabet_train/train')
    train_dir = os.path.join(PATH, 'augmented')
    test_dir = os.path.join(PATH, 'test')
    # test_dir = os.path.join(ROOT_DIR, 'Datasets/Larcc_dataset/test_ASL')

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=True,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE,
                                                                label_mode='categorical')

    test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                               shuffle=True,
                                                               batch_size=BATCH_SIZE,
                                                               image_size=IMG_SIZE,
                                                               label_mode='categorical')

    if not args["load_features"]:
        # dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
        #                                                       shuffle=True,
        #                                                       batch_size=BATCH_SIZE,
        #                                                       image_size=IMG_SIZE,
        #                                                       label_mode='categorical')



        # dataset_batches = tf.data.experimental.cardinality(dataset)
        # test_dataset = dataset.take(dataset_batches // 5) # test 20% of total
        # train_dataset = dataset.skip(int(dataset_batches // 5)) # Train is remaining 80%

        # print(test_dataset)

        class_names = train_dataset.class_names
        #
        # plt.figure(figsize=(10, 10))
        # for images, labels in dataset.take(1):
        #     for i in range(9):
        #         print(labels)
        #         ax = plt.subplot(3, 3, i + 1)
        #         plt.imshow(images[i].numpy().astype("uint8"))
        #         plt.title(class_names[labels[i]])
        #         plt.axis("off")

        # plt.show()

        # Split train and val 80/20

        train_batches = tf.data.experimental.cardinality(train_dataset)
        validation_dataset = train_dataset.take(train_batches // 5)
        # train_dataset = train_dataset.skip(train_batches // 5)

        print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
        print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

        # This helps with the training performance somehow
        AUTOTUNE = tf.data.AUTOTUNE

        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
        test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # <=================================================================================================>
    # <====================================DATASET AUGMENTATION=========================================>
    # <=================================================================================================>

    data_augmentation = tf.keras.Sequential([
        # tf.keras.layers.Resizing(224, 224, 'bilinear', False),
        # tf.keras.layers.RandomFlip('horizontal'),
        # tf.keras.layers.RandomRotation(0.1),
        # tf.keras.layers.RandomZoom((0.8, 1.5), None, "reflect", "bilinear")
    ])

    if not args["load_features"]:
        for image, _ in train_dataset.take(1):
            plt.figure(figsize=(10, 10))
            first_image = image[0]
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
                plt.imshow(augmented_image[0] / 255)
                plt.axis('off')

        # plt.show()

    # <=================================================================================================>
    # <=======================================MODEL DECISIONS===========================================>
    # <=================================================================================================>

    model_architecture = "InceptionV3"
    pooling = "MaxPooling"
    model_name = f"{model_architecture}_augmented2"
    training_epochs = 200
    training_batch_size = 2000
    training_patience = 10

    # <=================================================================================================>
    # <=======================================MODEL CREATION============================================>
    # <=================================================================================================>
    IMG_SHAPE = IMG_SIZE + (3,)
    val_split = 0.3

    if "ResNet50" in model_architecture:
        feature_extractor, features_input = create_resnet_base_model(IMG_SHAPE, pooling)
    elif "MobileNetV2" in model_architecture:
        feature_extractor, features_input = create_mobilenetv2_base_model(IMG_SHAPE, pooling)
    elif "InceptionV3" in model_architecture:
        feature_extractor, features_input = create_inceptionnet_base_model(IMG_SHAPE, pooling)
    else:
        feature_extractor, features_input = create_mobilenetv2_base_model(IMG_SHAPE, pooling)

    # Decision Model
    decision_input_shape = config[model_architecture][pooling]

    decision_inputs = tf.keras.Input(shape=(decision_input_shape,))
    x_d = tf.keras.layers.Dense(4)(decision_inputs)
    # x_d = tf.keras.activations.relu(x_d)
    # x_d = tf.keras.layers.Dropout(0.1)(x_d)
    # x_d = tf.keras.layers.Dense(128)(x_d)
    # x_d = tf.keras.activations.relu(x_d)
    # x_d = tf.keras.layers.Dense(4)(x_d)
    decision_outputs = tf.keras.activations.softmax(x_d)
    decision_model = tf.keras.Model(decision_inputs, decision_outputs)

    base_learning_rate = 0.0001

    decision_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

    feature_extractor.summary()
    decision_model.summary()

    # <=================================================================================================>
    # <===================================FEATURE EXTRACTION============================================>
    # <=================================================================================================>
    if not os.path.exists(f"/home/{USERNAME}/Datasets/extracted_features/{model_architecture}_{pooling}"):
        os.mkdir(f"/home/{USERNAME}/Datasets/extracted_features/{model_architecture}_{pooling}")

    st = time.time()

    extracted_features = None
    extracted_labels = None

    if args['load_features']:
        extracted_features = np.load(f"/home/{USERNAME}/Datasets/extracted_features"
                                     f"/{model_architecture}_{pooling}/extracted_features.npy")
        extracted_labels = np.load(f"/home/{USERNAME}/Datasets/extracted_features"
                                   f"/{model_architecture}_{pooling}/extracted_labels.npy")

        print(extracted_features.shape)
        print(extracted_labels.shape)

    if extracted_features is None or extracted_labels is None:

        extracted_features = None
        extracted_labels = None

        for i, batch in enumerate(iter(train_dataset)):

            image_batch = batch[0]
            label_batch = batch[1]

            # new_image_batch = tf.image.rgb_to_hsv(image_batch)

            # image_batch, label_batch = next(iter(test_dataset))
            feature_batch_average = feature_extractor(image_batch)
            # feature_batch_average = feature_extractor(new_image_batch)
            print(i)
            if extracted_features is None:
                extracted_features = feature_batch_average.numpy()
                extracted_labels = label_batch.numpy()
            else:
                extracted_features = np.concatenate((extracted_features, feature_batch_average.numpy()), axis=0)
                extracted_labels = np.concatenate((extracted_labels, label_batch.numpy()), axis=0)

            print(extracted_features.shape)
            print(extracted_labels.shape)

        np.save(f"/home/{USERNAME}/Datasets/extracted_features/{model_architecture}_{pooling}"
                f"/extracted_features.npy", extracted_features)
        np.save(f"/home/{USERNAME}/Datasets/extracted_features/{model_architecture}_{pooling}"
                f"/extracted_labels.npy", extracted_labels)

    extracted_features = tf.constant(extracted_features)
    extracted_labels = tf.constant(extracted_labels)

    feature_time = round(time.time() - st, 2)
    # <=================================================================================================>
    # <===================================MODEL TRAINING================================================>
    # <=================================================================================================>

    st = time.time()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=training_patience)

    history = decision_model.fit(x=extracted_features,
                                 y=extracted_labels,
                                 epochs=training_epochs,
                                 validation_split=val_split,
                                 shuffle=True,
                                 verbose=1,
                                 batch_size=training_batch_size,
                                 callbacks=[callback])

    training_time = round(time.time() - st, 2)

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
    # plt.show()

    # <=================================================================================================>
    # <===================================MODEL SAVING==================================================>
    # <=================================================================================================>

    outputs = decision_model(feature_extractor.outputs)

    trained_model = tf.keras.Model(feature_extractor.inputs, outputs)
    trained_model.summary()

    trained_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

    trained_model.save(ROOT_DIR + f"/hand_classification/network/{model_name}/myModel")
    # decision_model.save(ROOT_DIR + f"/hand_classification/network/{model_name}_decision/myModel")

    training_stats = {"train_samples": extracted_features.shape[0],
                      "val_samples": int(extracted_features.shape[0] * val_split),
                      "test_samples": int(tf.data.experimental.cardinality(test_dataset) * BATCH_SIZE),
                      "feature_extraction_time": feature_time,
                      "training_time": training_time}

    print(training_stats)

    user_data_json = json.dumps(training_stats, indent=4)

    with open(ROOT_DIR + f"/hand_classification/network/{model_name}/training_data.json", "w") as outfile:
        outfile.write(user_data_json)

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

    plt.show()



