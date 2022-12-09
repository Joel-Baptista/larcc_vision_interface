import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from vision_config.vision_definitions import ROOT_DIR
import copy
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Trains a new model based on a pre-trained model')
    parser.add_argument('-lf', '--load_features', action='store_true')
    args = vars(parser.parse_args())
    print(args)

    #<=================================================================================================>
    #<====================================DATASET PREPARATION==========================================>
    #<=================================================================================================>

    BATCH_SIZE = 300
    IMG_SIZE = (200, 200)

    if not args["load_features"]:
        PATH = ROOT_DIR + "/Datasets/ASL"

        train_dir = os.path.join(PATH, 'asl_alphabet_train/train')

        dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                              shuffle=True,
                                                              batch_size=BATCH_SIZE,
                                                              image_size=IMG_SIZE,
                                                              label_mode='categorical')

        dataset_batches = tf.data.experimental.cardinality(dataset)
        test_dataset = dataset.take(dataset_batches // 5) # Test 20% of total
        train_dataset = dataset.skip(int(dataset_batches // 5)) # Train is remaining 80%

        print(test_dataset)

        class_names = dataset.class_names
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
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
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
    # <=======================================MODEL CREATION============================================>
    # <=================================================================================================>

    # Mobilenet expects values between [-1, 1] instead of [0, 255]
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    # preprocess_input = tf.keras.applications.resnet50.preprocess_input
    model_name = "MobileNetV2"
    val_split = 0.3

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    # IMG_SHAPE = (224, 224, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    # base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
    #                                             include_top=False,
    #                                             weights='imagenet')

    base_model.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    # prediction_layer = tf.keras.layers.Dense(4)

    # Feature extracter model
    feature_inputs = tf.keras.Input(shape=IMG_SHAPE)
    x_f = data_augmentation(feature_inputs)
    x_f = preprocess_input(x_f)
    x_f = base_model(x_f, training=False)
    features = global_average_layer(x_f)
    feature_extractor = tf.keras.Model(feature_inputs, features)

    # Decision Model

    decision_inputs = tf.keras.Input(shape=(1280,))
    x_d = tf.keras.layers.Dense(4)(decision_inputs)
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

    st = time.time()

    extracted_features = None
    extracted_labels = None

    if args['load_features']:
        extracted_features_csv = pd.read_csv(ROOT_DIR + f"/Datasets/ASL/extracted_features"
                                                        f"/{model_name}/extracted_features.csv")
        extracted_labels_csv = pd.read_csv(ROOT_DIR + f"/Datasets/ASL/extracted_features"
                                                      f"/{model_name}/extracted_labels.csv")

        extracted_features = tf.constant(extracted_features_csv)
        extracted_labels = tf.constant(extracted_labels_csv)

        extracted_features = extracted_features[:, 1:]
        extracted_labels = extracted_labels[:, 1:]

        print(extracted_features.shape)
        print(extracted_labels.shape)

    if extracted_features is None or extracted_labels is None:

        extracted_features = None
        extracted_labels = None

        for i, batch in enumerate(iter(train_dataset)):
            image_batch = batch[0]
            label_batch = batch[1]
            # image_batch, label_batch = next(iter(test_dataset))
            feature_batch_average = feature_extractor(image_batch)
            print(i)
            if extracted_features is None:
                extracted_features = feature_batch_average
                extracted_labels = label_batch
            else:
                extracted_features = tf.concat(axis=0, values=[extracted_features, feature_batch_average])
                extracted_labels = tf.concat(axis=0, values=[extracted_labels, label_batch])

            print(extracted_features.shape)
            print(extracted_labels.shape)

        pd.DataFrame(np.array(extracted_features)).to_csv(
            ROOT_DIR + f"/Datasets/ASL/extracted_features/{model_name}/extracted_features.csv")
        pd.DataFrame(np.array(extracted_labels)).to_csv(
            ROOT_DIR + f"/Datasets/ASL/extracted_features/{model_name}/extracted_labels.csv")

    feature_time = round(time.time() - st, 2)
    # <=================================================================================================>
    # <===================================MODEL TRAINING================================================>
    # <=================================================================================================>

    st = time.time()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # history = model.fit(train_dataset,
    #                     epochs=initial_epochs,
    #                     validation_data=validation_dataset,
    #                     shuffle=True,
    #                     verbose=2,
    #                     batch_size=200,
    #                     callbacks=[callback])

    history = decision_model.fit(x=extracted_features,
                                 y=extracted_labels,
                                 epochs=500,
                                 validation_split=val_split,
                                 shuffle=True,
                                 verbose=2,
                                 batch_size=300,
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

    trained_model.save(ROOT_DIR + f"/hand_classification/network/{model_name}/myModel")

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

    count_true = 0
    count_false = 0
    ground_truth = []
    confusion_predictions = []
    gestures = ["A", "F", "L", "Y"]

    if not args["load_features"]:
        for i, batch in enumerate(iter(test_dataset)):
            # image_batch, label_batch = next(iter(test_dataset))
            image_batch = batch[0]
            label_batch = batch[1]
            feature_batch = feature_extractor(image_batch)

            predictions = decision_model(feature_batch)
            print(i)
            for j, prediction in enumerate(predictions):

                confusion_predictions.append(gestures[np.argmax(prediction)])
                ground_truth.append(gestures[np.argmax(label_batch[j])])

                if np.argmax(prediction) == np.argmax(label_batch[j]):
                    count_true += 1
                else:
                    count_false += 1

        cm = confusion_matrix(ground_truth, confusion_predictions, labels=gestures)
        blues = plt.cm.Blues
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gestures)
        disp.plot(cmap=blues)

        print(f"Accuracy: {round(count_true / (count_false + count_true) * 100, 2)}%")
        print(f"Tested with: {count_false + count_true}")

        plt.show()



