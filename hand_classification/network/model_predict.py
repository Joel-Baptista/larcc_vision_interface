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

dataset_path = ROOT_DIR + "/Datasets/Joel/"
data_path = f""

model = keras.models.load_model("myModel")

res = os.listdir(dataset_path + data_path)

im = cv2.imread(dataset_path + data_path + res[4])
pd = PoseDetection()

pd.cv_image = im
pd.detect_pose()
pd.find_hands()

cv2.imshow('Left Hand', pd.cv_image_detected_left)
cv2.imshow('Right Hand', pd.cv_image_detected_right)

im_array = np.asarray([pd.cv_image_detected_left, pd.cv_image_detected_right])

prediction = model.predict(x=im_array, verbose=2)
print(prediction)

cv2.imshow('Raw Image', im)
cv2.waitKey()
