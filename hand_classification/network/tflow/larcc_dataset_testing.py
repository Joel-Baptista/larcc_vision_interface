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
import pandas as pdd
from sklearn.mixture import GaussianMixture
from hand_classification.network.tflow.transfer_learning_funcs import *
from skimage.color import rgb2ycbcr, gray2rgb, rgb2yiq


def get_largest_item(mask):

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_mask = np.zeros(mask.shape)

    all_areas = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)

    largest_item = max(contours, key=cv2.contourArea)

    kernel = np.ones((3, 3), np.uint8)

    largest_mask = cv2.drawContours(largest_mask, [largest_item], -1, 255, thickness=-1)
    largest_mask = cv2.dilate(largest_mask, kernel)
    # largest_mask = cv2.fillPoly(largest_mask, pts=largest_item, color=255)
    # largest_mask = cv2.floodFill(largest_mask, largest_item, color=255)

    return largest_mask


def remove_bg(image, gmm1, gmm2):
    proc_image1 = rgb2ycbcr(image)[:, :, 1]
    proc_image2 = 255 * rgb2yiq(image)[:, :, 1]
    proc_image = np.reshape(cv2.merge([proc_image1, proc_image2]), (-1, 2))

    skin_score = gmm1.score_samples(proc_image)
    not_skin_score = gmm2.score_samples(proc_image)
    result = skin_score > not_skin_score

    result = result.reshape(image.shape[0], image.shape[1])

    mask2 = get_largest_item(result.astype(np.uint8))

    segmented_image = copy.deepcopy(image)
    segmented_image[mask2 == 0] = [0, 0, 0]

    return segmented_image


if __name__ == '__main__':

    df = pdd.read_csv(f'/home/{USERNAME}/Datasets/Skin_NonSkin.txt', header=None, delim_whitespace=True)
    df.columns = ['B', 'G', 'R', 'skin']
    df['Cb'] = np.round(128 - .168736 * df.R - .331364 * df.G +
                        .5 * df.B).astype(int)
    # df['Cr'] = np.round(128 + .5 * df.R - .418688 * df.G -
    #                     .081312 * df.B).astype(int)
    df['I'] = np.round(.596 * df.R - .275 * df.G -
                       .321 * df.B).astype(int)
    df.drop(['B', 'G', 'R'], axis=1, inplace=True)
    k = 4

    skin_data = df[df.skin == 1].drop(['skin'], axis=1).to_numpy()
    not_skin_data = df[df.skin == 2].drop(['skin'], axis=1).to_numpy()

    skin_gmm = GaussianMixture(n_components=k, covariance_type='full').fit(skin_data)
    not_skin_gmm = GaussianMixture(n_components=k, covariance_type='full').fit(not_skin_data)

    model_name = "InceptionV3_augmented2"

    model = keras.models.load_model(ROOT_DIR + f"/hand_classification/models/{model_name}/myModel")

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
    folder = "/home_testing"
    # folder = "/larcc_test_1"
    # folder = "/astra"
    # folder = "/larcc_test_1/blurred_33_33"

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

            # frame = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
            # cv2.imshow("Image", frame)
            # cv2.waitKey(5)
            # #
            # frame = cv2.flip(frame, 1)
            #
            # frame = cv2.GaussianBlur(frame, (7, 7), 0)

            # dic_test["gesture"].append(g)
            # dic_test["image name"].append(file)
            # buffer.append(np.array(frame))
            # ground_truth.append(np.array(ground_truth_array))
            # #
            cv2.imshow("Original", img)
            # pd.cv_image = copy.deepcopy(img)
            # pd.detect_pose()
            # pd.find_hands(x_lim=100, y_lim=100)

            pd.cv_image_detected_left = img

            if pd.cv_image_detected_left is not None:
                frame = img
                # frame = cv2.resize(pd.cv_image_detected_left, (200, 200), interpolation=cv2.INTER_CUBIC)

                # frame = remove_bg(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), skin_gmm, not_skin_gmm)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 21, 7, 21)

                # frame = cv2.flip(pd.cv_image_detected_right, 1)

                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                cv2.imshow("test", frame)
                cv2.waitKey(5)

                dic_test["gesture"].append(g)
                dic_test["image name"].append(file)
                buffer.append(np.array(frame))
                ground_truth.append(np.array(ground_truth_array))
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
        g = config[dataset]["gestures"][np.argmax(prediction)]
        confusion_predictions.append(config[dataset]["gestures"][np.argmax(prediction)])
        confusion_ground_truth.append(config[dataset]["gestures"][np.argmax(ground_truth[j])])

        if np.argmax(prediction) == np.argmax(ground_truth[j]):
            count_true += 1
        else:
            count_false += 1
            cv2.imwrite(f"/home/{USERNAME}/Datasets/Larcc_dataset/astra/wrong_classified/{g}/image{count_false}.png", buffer[j])

    print(f"Accuracy: {round(count_true / (count_false + count_true) * 100, 2)}%")
    print(f"Tested with: {count_false + count_true}")

    df = pdd.DataFrame(dic_test)
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


