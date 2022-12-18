import tensorflow as tf


def create_extraction_layer(extraction_type: str) -> tf.keras.layers:

    if extraction_type == "Flatten":
        return tf.keras.layers.Flatten()
    elif extraction_type == "MaxPooling":
        return tf.keras.layers.GlobalAveragePooling2D()

    return tf.keras.layers.GlobalAveragePooling2D()


def create_resnet_base_model(img_shape: tuple, extraction_type: str):

    preprocess_input = tf.keras.applications.resnet50.preprocess_input

    base_model = tf.keras.applications.ResNet50(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')

    base_model.trainable = False

    extraction_layer = create_extraction_layer(extraction_type)

    feature_inputs = tf.keras.Input(shape=img_shape)
    # x_f = data_augmentation(feature_inputs)
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
    # x_f = data_augmentation(feature_inputs)
    x_f = preprocess_input(feature_inputs)
    x_f = base_model(x_f, training=False)
    features = extraction_layer(x_f)
    extractor = tf.keras.Model(feature_inputs, features)

    return extractor, feature_inputs

