import argparse
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from hand_classification.network.tflow.transfer_learning import TransferLearning
from hand_classification.network.tflow.transfer_learning_funcs import plot_curves, plot_confusion, test_model
from vision_config.vision_definitions import ROOT_DIR, USERNAME


class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


def add_projection_head(encoder):
    inputs = keras.Input(shape=(2048,))
    # features = encoder(inputs)
    x = layers.Dense(1024, activation="relu")(inputs)
    x = layers.Dense(512, activation="relu")(x)
    # x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(256, activation="relu")(x)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model


def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=(2048,))
    features = encoder(inputs)
    # features = layers.Dropout(0.1)(features)
    features = layers.Dense(64, activation="relu")(features)
    # features = layers.Dropout(0.1)(features)
    outputs = layers.Dense(4, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trains a new model based on a pre-trained model')
    parser.add_argument('-lf', '--load_features', action='store_true')
    args = vars(parser.parse_args())

    folder = "larcc_test_1/detected"

    transfer_learning = TransferLearning(ROOT_DIR + '/hand_classification/config/transfer_learning.json')

    transfer_learning.base_model = "InceptionV3"
    transfer_learning.model_name = f"{transfer_learning.base_model}_larcc_contrastive_256"
    transfer_learning.IMG_SIZE = (224, 224)

    model_freeze, model_train = transfer_learning.create_model()

    train_data, test_data = \
        transfer_learning.load_data(train_dir_path=f"/home/{USERNAME}/Datasets/ASL/train_kinect",
                                    test_dir_path=f"/home/{USERNAME}/Datasets/Larcc_dataset/{folder}")

    features_train, labels_train = transfer_learning.get_features(model_freeze, train_data, args['load_features'])

    encoder_with_projection_head = add_projection_head(model_freeze)
    encoder_with_projection_head.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=SupervisedContrastiveLoss(0.05)
    )

    encoder_with_projection_head.summary()

    labels_categorical = []

    for label in labels_train:

        vector = [0, 0, 0, 0]
        vector[np.array(label)] = 1

        labels_categorical.append(vector)

    print(np.array(labels_categorical))

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    callback2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = encoder_with_projection_head.fit(
        x=features_train, y=labels_train, batch_size=256, epochs=1000, verbose=2,
        callbacks=[callback]
    )

    classifier = create_classifier(encoder_with_projection_head, trainable=False)

    history_classifier = classifier.fit(x=features_train, y=np.array(labels_categorical), batch_size=2000, epochs=1000,
                             verbose=2,
                             validation_split = 0.3,
                             callbacks=[callback2], 
                             )

    # inputs = keras.Input(shape=(200, 200, 3,))
    # features = model_freeze(inputs)
    # compressed_features = encoder_with_projection_head(features)
    # outputs = classifier(compressed_features)

    inputs = keras.Input(shape=(224, 224, 3, ))
    features = model_freeze(inputs)
    outputs = classifier(features)

    model = keras.Model(
        inputs=inputs, outputs=outputs
    )

    model.save(ROOT_DIR + f"/hand_classification/models/{transfer_learning.model_name}/myModel")

    # accuracy = classifier.evaluate(x_test, y_test)[1]
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    # trained_model = transfer_learning.save_model(model_freeze, model_train)
    #
    plot_curves(history_classifier)
    #
    ground_truth, predictions, results = test_model(model, test_data, ["A", "F", "L", "Y"],
                                                    folder, transfer_learning.model_name, "Larcc_dataset")
    
    # plot_confusion(ground_truth, predictions, ["A", "F", "L", "Y"])