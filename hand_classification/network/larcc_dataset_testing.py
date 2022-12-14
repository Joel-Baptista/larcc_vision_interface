#!/usr/bin/env python3
from hand_detection.src.pose_detection import PoseDetection
from vision_config.vision_definitions import ROOT_DIR, USERNAME
import json
import cv2
import numpy as np
import keras
import os
import copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, \
    precision_score, f1_score
import matplotlib.pyplot as plt
import pandas
from hand_classification.network.transfer_learning_funcs import *

if __name__ == '__main__':

    model_name = "InceptionV3_blurred"

    model = keras.models.load_model(ROOT_DIR + f"/hand_classification/network/{model_name}/myModel")

    dataset = "ASL"

    with open(f'{ROOT_DIR}/Dataset/configs/larcc_dataset_config.json') as f:
        config = json.load(f)

    pd = PoseDetection()
    dic_test = {"gesture": [],
                "image name": [],
                "prediction": [],
                # "probabilities": []
                }

    # folder = ""
    # folder = "/home_testing"
    # folder = "/larcc_test_1"
    folder = "/larcc_test_1/blurred_33_33"

    total_images = 0
    for g in config[dataset]["gestures"]:
        res = os.listdir(f"/home/{USERNAME}/Datasets/Larcc_dataset{folder}/{g}")
        total_images += len(res)

    print(f"There are {total_images} images in this dataset")
    buffer = []
    ground_truth = []
    ground_truth_index = 0
    count = 0
    last_percentage = 0
    for g in config[dataset]["gestures"]:

        res = os.listdir(f"/home/{USERNAME}/Datasets/Larcc_dataset{folder}/{g}")
        ground_truth_array = [0] * len(config[dataset]["gestures"])
        ground_truth_array[ground_truth_index] = 1
        for file in res:
            if (count / total_images) * 100 >= last_percentage:
                print(f"{last_percentage}% of images analysed")
                last_percentage += 10

            count += 1
            img = cv2.imread(f"/home/{USERNAME}/Datasets/Larcc_dataset{folder}/{g}/{file}")

            frame = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Image", frame)
            cv2.waitKey(5)
            #
            # frame = cv2.flip(frame, 1)
            #
            # frame = cv2.GaussianBlur(frame, (7, 7), 0)

            dic_test["gesture"].append(g)
            dic_test["image name"].append(file)
            buffer.append(np.array(frame))
            ground_truth.append(np.array(ground_truth_array))
            #
            # cv2.imshow("Original", img)
            # pd.cv_image = copy.deepcopy(img)
            # pd.detect_pose()
            # pd.find_hands(x_lim=75, y_lim=75)

            # if pd.cv_image_detected_left is not None:
            #     frame = cv2.resize(pd.cv_image_detected_left, (200, 200), interpolation=cv2.INTER_CUBIC)
            #     #
            #     frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 21, 7, 21)
            #
            #     # frame = cv2.flip(pd.cv_image_detected_right, 1)
            #
            #     # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #     cv2.imshow("test", frame)
            #     cv2.waitKey(5)
            #
            #     dic_test["gesture"].append(g)
            #     dic_test["image name"].append(file)
            #     buffer.append(np.array(frame))
            #     ground_truth.append(np.array(ground_truth_array))
            #     cv2.imwrite(f"/home/{USERNAME}/Datasets/Larcc_dataset/testing/{g}/image{count}.png", frame)

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
    print(config[dataset]["gestures"])

    # print(np.array(predictions))
    # print(np.array(ground_truth))

    count_true = 0
    count_false = 0
    confusion_predictions = []
    confusion_ground_truth = []
    for j, prediction in enumerate(predictions):
        dic_test["prediction"].append(config[dataset]["gestures"][np.argmax(prediction)])

        # dic_test["probabilities"].append(str(prediction))

        confusion_predictions.append(config[dataset]["gestures"][np.argmax(prediction)])
        confusion_ground_truth.append(config[dataset]["gestures"][np.argmax(ground_truth[j])])

        if np.argmax(prediction) == np.argmax(ground_truth[j]):
            count_true += 1
        else:
            count_false += 1

    print(f"Accuracy: {round(count_true / (count_false + count_true) * 100, 2)}%")
    print(f"Tested with: {count_false + count_true}")

    df = pandas.DataFrame(dic_test)
    df.to_csv(f"/home/{USERNAME}/Datasets/Larcc_dataset{folder}/{model_name}_test_results.csv")

    recall = recall_score(confusion_ground_truth, confusion_predictions, average=None)
    precision = precision_score(confusion_ground_truth, confusion_predictions, average=None)
    f1 = f1_score(confusion_ground_truth, confusion_predictions, average=None)

    dic_results = {"accuracy": accuracy_score(confusion_ground_truth, confusion_predictions), "precision": {},
                   "recall": {}, "f1": {}}

    for i, label in enumerate(config[dataset]["gestures"]):
        print(i)
        dic_results["recall"][label] = recall[i]
        dic_results["precision"][label] = precision[i]
        dic_results["f1"][label] = f1[i]

    dic_results["recall"]["average"] = np.mean(recall)
    dic_results["precision"]["average"] = np.mean(precision)
    dic_results["f1"]["average"] = np.mean(f1)

    print(accuracy_score(confusion_ground_truth, confusion_predictions))
    print(recall_score(confusion_ground_truth, confusion_predictions, average=None))
    print(precision_score(confusion_ground_truth, confusion_predictions, average=None))
    print(f1_score(confusion_ground_truth, confusion_predictions, average=None))

    print(dic_results)

    with open(f"/home/{USERNAME}/Datasets/Larcc_dataset{folder}/{model_name}_results.json", 'w') as outfile:
        json.dump(dic_results, outfile)

    cm = confusion_matrix(confusion_ground_truth, confusion_predictions, labels=config[dataset]["gestures"])
    blues = plt.cm.Blues
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config[dataset]["gestures"])
    disp.plot(cmap=blues)

    plt.savefig(f"/home/{USERNAME}/Datasets/Larcc_dataset{folder}/cm.png")

    plt.show()


