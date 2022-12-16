#!/usr/bin/env python3
from hand_detection.src.pose_detection import PoseDetection
from vision_config.vision_definitions import ROOT_DIR
import json
import cv2
import numpy as np
import keras
import os
import copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas

if __name__ == '__main__':

    model = keras.models.load_model(ROOT_DIR + f"/hand_classification/network/ResNet50_ASL1/myModel")

    with open(f'{ROOT_DIR}/Datasets/Larcc_dataset/larcc_dataset_config.json') as f:
        config = json.load(f)

    pd = PoseDetection()
    dic_test = {"gesture": [],
                "image name": [],
                "prediction": [],
                # "probabilities": []
                }

    total_images = 0
    for g in config["gestures"]:
        res = os.listdir(ROOT_DIR + f"/Datasets/Larcc_dataset/{g}")
        total_images += len(res)

    print(f"There are {total_images} images in this dataset")
    buffer = []
    ground_truth = []
    ground_truth_index = 0
    count = 0
    last_percentage = 0
    for g in config["gestures"]:

        res = os.listdir(ROOT_DIR + f"/Datasets/Larcc_dataset/{g}")
        ground_truth_array = [0] * len(config["gestures"])
        ground_truth_array[ground_truth_index] = 1
        for file in res:
            if (count / total_images) * 100 >= last_percentage:
                print(f"{last_percentage}% of images analysed")
                last_percentage += 10

            count += 1
            img = cv2.imread(ROOT_DIR + f"/Datasets/Larcc_dataset/{g}/{file}")

            pd.cv_image = copy.deepcopy(img)
            pd.detect_pose()
            pd.find_hands(x_lim=100, y_lim=100)

            frame = cv2.resize(pd.cv_image_detected_left, (200, 200), interpolation=cv2.INTER_CUBIC)

            cv2.imshow("test", frame)
            cv2.waitKey(5)
            if pd.cv_image_detected_left is not None:

                dic_test["gesture"].append(g)
                dic_test["image name"].append(file)
                buffer.append(np.array(frame))
                ground_truth.append(np.array(ground_truth_array))
                # cv2.imwrite(f"{ROOT_DIR}/Datasets/Larcc_dataset/Testing/{g}{file}", pd.cv_image_detected_left)

            # if pd.cv_image_detected_left is not None:
            #
            #     dic_test["gesture"].append(g)
            #     dic_test["image name"].append(file)
            #     buffer.append(np.array(pd.cv_image_detected_left))
            #     ground_truth.append(np.array(ground_truth_array))
            #     # cv2.imwrite(f"{ROOT_DIR}/Datasets/Larcc_dataset/Testing/{g}{file}", pd.cv_image_detected_left)

        ground_truth_index += 1

    print(np.array(ground_truth).shape)
    print(np.array(buffer).shape)

    predictions = model.predict(x=np.array(buffer), verbose=2)

    print(predictions.shape)
    print(config["gestures"])

    # print(np.array(predictions))
    # print(np.array(ground_truth))

    count_true = 0
    count_false = 0
    confusion_predictions = []
    confusion_ground_truth = []
    for j, prediction in enumerate(predictions):
        dic_test["prediction"].append(config["gestures"][np.argmax(prediction)])

        # dic_test["probabilities"].append(str(prediction))

        confusion_predictions.append(config["gestures"][np.argmax(prediction)])
        confusion_ground_truth.append(config["gestures"][np.argmax(ground_truth[j])])

        if np.argmax(prediction) == np.argmax(ground_truth[j]):
            count_true += 1
        else:
            count_false += 1

    print(f"Accuracy: {round(count_true / (count_false + count_true) * 100, 2)}%")
    print(f"Tested with: {count_false + count_true}")

    df = pandas.DataFrame(dic_test)
    df.to_csv(f"{ROOT_DIR}/Datasets/Larcc_dataset/Testing/test_results.csv")

    cm = confusion_matrix(confusion_ground_truth, confusion_predictions, labels=config["gestures"])
    blues = plt.cm.Blues
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config["gestures"])
    disp.plot(cmap=blues)

    plt.show()
