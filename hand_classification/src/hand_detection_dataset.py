#!/usr/bin/env python3
import os

import cv2
from vision_config.vision_definitions import ROOT_DIR
from hand_detection.src.hand_detection import HandDetection
from hand_detection.src.pose_detection import PoseDetection


dataset_path = "/Datasets/HANDS_dataset"

subject = 2
gesture = 1

subject_path = f"/Subject{subject}/Subject{subject}/G{gesture}/"

res = os.listdir(ROOT_DIR + dataset_path + subject_path)

# full_path = "/home/joel/Imagens/hands3.jpg"
pd = PoseDetection()

for file in res:

    im = cv2.imread(ROOT_DIR + dataset_path + subject_path + file)

    pd.cv_image = im
    pd.detect_pose()
    pd.find_hands()

    cv2.imshow('Original Images', im)
    cv2.imshow('Left Hand', pd.cv_image_detected_left)
    cv2.imshow('Right Hand', pd.cv_image_detected_right)

    key = cv2.waitKey(500)

    if key == 113:
        break
