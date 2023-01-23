#!/usr/bin/env python3
from vision_config.vision_definitions import ROOT_DIR, USERNAME
import os
import json
import imgaug.augmenters as iaa
import imgaug as ia
import numpy as np
import cv2


if __name__ == '__main__':

    gestures = ["A", "F", "L", "Y"]

    bgs = os.listdir(f"/home/{USERNAME}/Datasets/test_dataset/kinect/wrong_classified/backgrounds")

    count = 0
    for g in gestures:
        res = os.listdir(f"/home/{USERNAME}/Datasets/test_dataset/kinect/wrong_classified/bg_removed/{g}")

        for file in res:

            img = cv2.imread(f"/home/{USERNAME}/Datasets/test_dataset/kinect/wrong_classified/bg_removed/{g}/{file}")

            mask = cv2.inRange(img, (0, 0, 0), (0, 0, 0))

            for bg in bgs:

                img_bg = cv2.imread(f"/home/{USERNAME}/Datasets/test_dataset/kinect/wrong_classified/backgrounds/{bg}")

                img_bg[mask == 0] = img[mask == 0]

                cv2.imwrite(f"/home/{USERNAME}/Datasets/test_dataset/kinect/"
                            f"wrong_classified/bg_changed/{g}/{bg}_{file}", img_bg)
