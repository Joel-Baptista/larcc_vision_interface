#!/usr/bin/env python3
import os

import keras
import numpy as np
import cv2

from vision_config.vision_definitions import ROOT_DIR

subject = 1
gesture = 3

dataset_path = ROOT_DIR + "/Datasets/HANDS_dataset"
data_path = f"/Subject{subject}/Processed/G{gesture}/"

model = keras.models.load_model("myModel")

res = os.listdir(dataset_path + data_path)

im = cv2.imread(dataset_path + data_path + res[0])

im_array = np.asarray([im])

prediction = model.predict(x=im_array, verbose=2)
print(prediction)

