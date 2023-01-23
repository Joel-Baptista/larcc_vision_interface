#!/usr/bin/env python3
import copy
import os
import cv2
from vision_config.vision_definitions import USERNAME
from hand_detection.src.pose_detection import PoseDetection
from dataset_funcs import *

if __name__ == '__main__':

    processess = ["gmm remove background", f'/home/{USERNAME}/Datasets/Skin_NonSkin.txt']
    # processess = []
    gestures = ["A", "F", "L", "Y"]
    folder = "kinect"
    process = "bg_removed"

    for i in range(0, len(gestures)):
        if not os.path.exists(f"/home/{USERNAME}/Datasets/test_dataset/{folder}/{process}/{gestures[i]}"):
            os.makedirs(f"/home/{USERNAME}/Datasets/test_dataset/{folder}/{process}/{gestures[i]}")

    total_images = 0
    for g in gestures:
        res = os.listdir(f"/home/{USERNAME}/Datasets/Larcc_dataset/{folder}/{g}")
        total_images += len(res)

    print(f"There are {total_images} images in this dataset")
    buffer = []
    ground_truth = []
    ground_truth_index = 0
    count = 0
    last_percentage = 0
    changing_index = []
    image_index = 0

    pd = PoseDetection(static_image_mode=True)

    print("Detecting Hands")
    for g in gestures:

        res = os.listdir(f"/home/{USERNAME}/Datasets/Larcc_dataset/{folder}/{g}")

        for file in res:
            if (count / total_images) * 100 >= last_percentage:
                print(f"{last_percentage}% of images analysed")
                last_percentage += 10

            count += 1

            img = cv2.imread(f"/home/{USERNAME}/Datasets/Larcc_dataset/{folder}/{g}/{file}")

            pd.cv_image = copy.deepcopy(img)
            pd.detect_pose()
            pd.find_hands(x_lim=100, y_lim=100)

            if pd.cv_image_detected_left is not None:
                buffer.append(pd.cv_image_detected_left)
                image_index += 1
            # if pd.cv_image_detected_right is not None:
            #     buffer.append(cv2.flip(pd.cv_image_detected_right, 1))
            #     image_index += 1

        changing_index.append(image_index)

    processed_images = preprocessing(buffer, processess)
    print(changing_index)
    gesture_index = 0
    file_count = 0
    for image in processed_images:

        if changing_index[gesture_index] == file_count:
            gesture_index += 1

        cv2.imwrite(f"/home/{USERNAME}/Datasets/test_dataset/{folder}/{process}/"
                    f"{gestures[gesture_index]}/image{file_count}.png", image)

        file_count += 1
















