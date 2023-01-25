from tensorflow import keras
import tensorflow as tf
from hand_classification.network.transfer_learning_funcs import test_model, plot_confusion
from vision_config.vision_definitions import ROOT_DIR, USERNAME

if __name__ == '__main__':

    model_name = "InceptionV3"
    folder = "/test"
    dataset = "ASL"

    test_dir_path = f"/home/{USERNAME}/Datasets/{dataset}/{folder}"
    model = keras.models.load_model(ROOT_DIR + f"/hand_classification/models/{model_name}/myModel")

    test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir_path,
                                                               shuffle=False,
                                                               batch_size=200,
                                                               image_size=(200, 200),
                                                               label_mode='categorical',
                                                               color_mode='rgb')

    ground_truth, predictions, results = test_model(model, test_dataset, ["A", "F", "L", "Y"],
                                                    folder, model_name, dataset)

    plot_confusion(ground_truth, predictions, ["A", "F", "L", "Y"])

