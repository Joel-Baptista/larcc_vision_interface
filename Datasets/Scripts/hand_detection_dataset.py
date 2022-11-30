#!/usr/bin/env python3
import os

import cv2
from vision_config.vision_definitions import ROOT_DIR
from hand_detection.src.hand_detection import HandDetection
from hand_detection.src.pose_detection import PoseDetection


dataset_path = "/Datasets/HANDS_dataset"

subjects = 5
gestures = 3

for subject in range(1, subjects + 1):
    if not os.path.exists(ROOT_DIR + dataset_path + f"/Subject{subject}"):
        os.makedirs(ROOT_DIR + dataset_path + f"/Subject{subject}/raw_images")
        os.makedirs(ROOT_DIR + dataset_path + f"/Subject{subject}/hand_segmented")

    for gesture in range(1, gestures + 1):
        if not os.path.exists(ROOT_DIR + dataset_path + f"/Subject{subject}/raw_images/G{gesture}"):
            os.makedirs(ROOT_DIR + dataset_path + f"/Subject{subject}/raw_images/G{gesture}")


# full_path = "/home/joel/Imagens/hands3.jpg"
pd = PoseDetection(static_image_mode=True)

for subject in range(1, subjects + 1):
    for gesture in range(1, gestures + 1):

        print(f"Processing gesture {gesture} from user {subject}")

        subject_path = f"/Subject{subject}/raw_images/G{gesture}/"
        res = os.listdir(ROOT_DIR + dataset_path + subject_path)
        saving_path = f"/Subject{subject}/hand_segmented/G{gesture}/"

        print(f"There is {len(res)} images to process")
        last_checkpoint = 0.0

        for i, file in enumerate(res):

            if i / len(res) > last_checkpoint:
                print(str(round(last_checkpoint * 100, 1)) + "%")
                last_checkpoint += 0.1

            im = cv2.imread(ROOT_DIR + dataset_path + subject_path + file)

            pd.cv_image = im
            pd.detect_pose()
            pd.find_hands()

            if (pd.cv_image_detected_left is not None) and (pd.cv_image_detected_right is not None):
                left_hand = cv2.resize(pd.cv_image_detected_left, (100, 100), interpolation=cv2.INTER_LINEAR_EXACT)
                right_hand = cv2.resize(pd.cv_image_detected_right, (100, 100), interpolation=cv2.INTER_LINEAR_EXACT)

                cv2.imwrite(ROOT_DIR + dataset_path + saving_path + file[:-4] + "_left.png", left_hand)
                cv2.imwrite(ROOT_DIR + dataset_path + saving_path + file[:-4] + "_right.png", right_hand)


            # cv2.imwrite(ROOT_DIR + dataset_path + saving_path + file[:-4] + "_complete.png", pd.cv_image_detected)

            # cv2.imshow('Detected Images', pd.cv_image_detected)
            # cv2.imshow('Original Images', im)
            # cv2.imshow('Left Hand', pd.cv_image_detected_left)
            # cv2.imshow('Right Hand', pd.cv_image_detected_right)

            key = cv2.waitKey()

            if key == 113:
                break
