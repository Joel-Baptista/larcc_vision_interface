#!/usr/bin/env python3
import os
from hand_detection.src.pose_detection import PoseDetection
import keras
import numpy as np
import cv2

from vision_config.vision_definitions import ROOT_DIR

subject = 1
gesture = 3

# dataset_path = ROOT_DIR + "/Datasets/HANDS_dataset"
# data_path = f"/Subject{subject}/Processed/G{gesture}/"

dataset_path = ROOT_DIR + "/Datasets/Joel_v2/Processed/"
data_path = f""

model = keras.models.load_model("myModel_second")
count_false = 0
count_true = 0
for gesture in range(1, 4):
    res = os.listdir(dataset_path + f"G{gesture}")
    true_value = 0
    for file in res:
        im = cv2.imread(dataset_path + f"G{gesture}/" + file)
        # pd = PoseDetection()
        #
        # pd.cv_image = im
        # pd.detect_pose()
        # pd.find_hands()

        # cv2.imshow('Left Hand', pd.cv_image_detected_left)
        # cv2.imshow('Right Hand', pd.cv_image_detected_right)

        im_array = np.asarray([im])

        prediction = model.predict(x=im_array, verbose=2)
        if gesture == (np.argmax(prediction) + 1):
            print("True")
            count_true += 1
        else:
            print("False")
            count_false += 1


print(f"Accuracy: {round(count_true / (count_false + count_true) * 100, 2)}%")
print(f"Tested with: {count_false + count_true}")
