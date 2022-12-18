import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from vision_config.vision_definitions import ROOT_DIR


if __name__ == '__main__':

    #<=================================================================================================>
    #<====================================DATASET PREPARATION==========================================>
    #<=================================================================================================>



    # _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    # _URL = ROOT_DIR + '/ASL'
    # path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
    # PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    PATH = ROOT_DIR + "/Datasets/ASL"

    train_dir = os.path.join(PATH, 'asl_alphabet_train/train')
    # validation_dir = os.path.join(PATH, 'asl_alphabet_test/asl_alphabet_test')

    BATCH_SIZE = 300
    IMG_SIZE = (200, 200)

    dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                          shuffle=True,
                                                          batch_size=BATCH_SIZE,
                                                          image_size=IMG_SIZE,
                                                          label_mode='categorical',
                                                          seed=88740)

    dataset_batches = tf.data.experimental.cardinality(dataset)
    test_dataset = dataset.take(dataset_batches // 5) # test 20% of total
    train_dataset = dataset.skip(int(dataset_batches // 5)) # Train is remaining 80%

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
    train_dataset = train_dataset.skip(train_batches // 5)

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
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])

    for image, _ in train_dataset.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')

    # plt.show()

    # Mobilenet expects values between [-1, 1] instead of [0, 255]
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # <=================================================================================================>
    # <=======================================MODEL CREATION============================================>
    # <=================================================================================================>

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average)
    feature_batch = base_model(image_batch)
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average)

    extracted_features = None
    extracted_labels = None

    for i, batch in enumerate(iter(train_dataset)):
        image_batch, label_batch = next(iter(train_dataset))
        feature_batch = base_model(image_batch)
        feature_batch_average = global_average_layer(feature_batch)
        print(i)
        if extracted_features is None:
            extracted_features = feature_batch_average
            extracted_labels = label_batch
        else:
            extracted_features = tf.concat(axis=0, values=[extracted_features, feature_batch_average])
            extracted_labels = tf.concat(axis=0, values=[extracted_labels, label_batch])

            print(extracted_features.shape)
            print(extracted_labels.shape)

    print(image_batch.shape)
    print(label_batch.shape)

    base_model.trainable = False

    tf.concat(axis=1, values=[tf.cast(tensor1, tf.float32), tensor2])
    # Let's take a look at the base model architecture
    # base_model.summary()

    feature_batch_average = global_average_layer(feature_batch)
    # print(feature_batch_average.shape)
    print(type(feature_batch_average))
    print(type(label_batch))

    prediction_layer = tf.keras.layers.Dense(4)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)
    print(prediction_batch)








