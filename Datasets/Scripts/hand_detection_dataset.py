#!/usr/bin/env python3
import os
import json
import cv2
from vision_config.vision_definitions import ROOT_DIR
from hand_detection.src.hand_detection import HandDetection
from hand_detection.src.pose_detection import PoseDetection
import re

dataset_path = "/Datasets/HANDS"

subjects = 5
gestures = ["G1", "G2", "G5", "G6"]
user_data = {}
test_user = 5

for i in range(0, len(gestures)):
    if not os.path.exists(ROOT_DIR + dataset_path + f"/train/{gestures[i]}"):
        os.makedirs(ROOT_DIR + dataset_path + f"/train/{gestures[i]}")
    if not os.path.exists(ROOT_DIR + dataset_path + f"/test/{gestures[i]}"):
        os.makedirs(ROOT_DIR + dataset_path + f"/test/{gestures[i]}")


f = open(ROOT_DIR + '/Datasets/HANDS/config.json')
images_indexes = json.load(f)
f.close()

pd = PoseDetection(static_image_mode=True)

# raw_dataset_path = "/home/joel/Desktop/Dataset"
raw_dataset_path = "/home/joelbaptista/Desktop/Hands_Dataset"

if os.path.exists(raw_dataset_path):
    for subject in range(1, subjects + 1):
        res = os.listdir(raw_dataset_path + f"/Subject{subject}")

        if subject == test_user:
            path = "test"
        else:
            path = "train"

        print(f"Processing subject {subject} out of {subjects} subjects")
        last_checkpoint = 0.0

        for i, file in enumerate(res):

            image_index = re.findall(r'\d+', file)

            if i / len(res) > last_checkpoint:
                print(str(round(last_checkpoint * 100, 1)) + "%")
                last_checkpoint += 0.1

            for G in images_indexes[f"user{subject}"]:
                ranges = images_indexes[f"user{subject}"][G]
                if ranges[0] <= int(image_index[0]) <= ranges[1]:

                    saving_path = f"/{path}/{G}/"

                    im = cv2.imread(raw_dataset_path + f"/Subject{subject}/{file}")

                    pd.cv_image = im
                    pd.detect_pose()
                    pd.find_hands(x_lim=100,
                                  y_lim=100)

                    if (pd.cv_image_detected_left is None) or (pd.cv_image_detected_right is None):
                        continue

                    if not (G in gestures):
                        continue

                    # user_data[f"user{subject}"][G] += 1

                    # left_hand = cv2.resize(pd.cv_image_detected_left, (224, 224), interpolation=cv2.INTER_LINEAR_EXACT)
                    # right_hand = cv2.resize(pd.cv_image_detected_right, (224, 224), interpolation=cv2.INTER_LINEAR_EXACT)

                    left_hand = pd.cv_image_detected_left
                    right_hand = pd.cv_image_detected_right

                    cv2.imwrite(ROOT_DIR + dataset_path + saving_path + f"{G}_" + file[:-4] + "_left.png", left_hand)
                    # cv2.imwrite(ROOT_DIR + dataset_path + saving_path + f"{G}_" + file[:-4] + "_right.png", right_hand)

                    # cv2.imshow('Detected Images', pd.cv_image_detected)
                    #
                    # key = cv2.waitKey(5)
                    #
                    # if key == 113:
                    #     break


print(user_data)
user_data_json = json.dumps(user_data, indent=4)

with open(ROOT_DIR + "/Datasets/HANDS_dataset/user_data.json", "w") as outfile:
    outfile.write(user_data_json)

# for subject in range(1, subjects + 1):
#     for gesture in range(1, gestures + 1):
#
#         print(f"Processing gesture {gesture} from user {subject}")
#
#         subject_path = f"/Subject{subject}/raw_images/G{gesture}/"
#         res = os.listdir(ROOT_DIR + dataset_path + subject_path)
#         saving_path = f"/Subject{subject}/hand_segmented/G{gesture}/"
#
#         print(f"There is {len(res)} images to process")
#         last_checkpoint = 0.0
#
#         for i, file in enumerate(res):
#
#             if i / len(res) > last_checkpoint:
#                 print(str(round(last_checkpoint * 100, 1)) + "%")
#                 last_checkpoint += 0.1
#
#             im = cv2.imread(ROOT_DIR + dataset_path + subject_path + file)
#
#             pd.cv_image = im
#             pd.detect_pose()
#             pd.find_hands()
#
#             if (pd.cv_image_detected_left is not None) and (pd.cv_image_detected_right is not None):
#                 left_hand = cv2.resize(pd.cv_image_detected_left, (100, 100), interpolation=cv2.INTER_LINEAR_EXACT)
#                 right_hand = cv2.resize(pd.cv_image_detected_right, (100, 100), interpolation=cv2.INTER_LINEAR_EXACT)
#
#                 cv2.imwrite(ROOT_DIR + dataset_path + saving_path + file[:-4] + "_left.png", left_hand)
#                 cv2.imwrite(ROOT_DIR + dataset_path + saving_path + file[:-4] + "_right.png", right_hand)
#
#
#             # cv2.imwrite(ROOT_DIR + dataset_path + saving_path + file[:-4] + "_complete.png", pd.cv_image_detected)
#
#             # cv2.imshow('Detected Images', pd.cv_image_detected)
#             # cv2.imshow('Original Images', im)
#             # cv2.imshow('Left Hand', pd.cv_image_detected_left)
#             # cv2.imshow('Right Hand', pd.cv_image_detected_right)
#
#             key = cv2.waitKey()
#
#             if key == 113:
#                 break
