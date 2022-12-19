import tensorflow as tf


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
    '''
    Input:
      model: A pre-trained Keras Sequential model
      index: The index of the layer where we want to split the model
    Output:
      model1: From layer 0 to index
      model2: From index+1 layer to the output of the original model
    The index layer will be the last layer of the model_1 and the same shape of that layer will be the input layer of the model_2
    '''

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

