#!/usr/bin/env python3
import os

import cv2
from vision_config.vision_definitions import ROOT_DIR
from hand_detection.src.hand_detection import HandDetection
from hand_detection.src.pose_detection import PoseDetection


dataset_path = "/Datasets/Joel_v2/"

subject = 3
gesture = 1

# full_path = "/home/joel/Imagens/hands3.jpg"
pd = PoseDetection()
#
# for subject in range(1, 6):
#     for gesture in range(1, 4):
#
#         print(f"Processing gesture {gesture} from user {subject}")
#
#         subject_path = f"/Subject{subject}/Subject{subject}/G{gesture}/"
#         res = os.listdir(ROOT_DIR + dataset_path + subject_path)
#         saving_path = f"/Subject{subject}/Processed/G{gesture}/"
#
#         print(f"There is {len(res)} images to process")
last_checkpoint = 0.0

res = os.listdir(ROOT_DIR + dataset_path + "/Raw/")
saving_path = ""
for i, file in enumerate(res):

    if i / len(res) > last_checkpoint:
        print(str(round(last_checkpoint * 100, 1)) + "%")
        last_checkpoint += 0.1

    if "gesture1" in file:
        saving_path = "/Processed/G1/"
    elif "gesture2" in file:
        saving_path = "/Processed/G2/"
    elif "gesture3" in file:
        saving_path = "/Processed/G3/"

    im = cv2.imread(ROOT_DIR + dataset_path+ "/Raw/" + file)

    pd.cv_image = im
    pd.detect_pose()
    pd.find_hands()

    cv2.imwrite(ROOT_DIR + dataset_path + saving_path + file[:-4] + "_left.png", pd.cv_image_detected_left)
    cv2.imwrite(ROOT_DIR + dataset_path + saving_path + file[:-4] + "_right.png", pd.cv_image_detected_right)

    # cv2.imshow('Original Images', im)
    # cv2.imshow('Left Hand', pd.cv_image_detected_left)
    # cv2.imshow('Right Hand', pd.cv_image_detected_right)
    #
    # key = cv2.waitKey(500)
    #
    # if key == 113:
    #     break
