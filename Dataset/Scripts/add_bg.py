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
    path = f"/home/{USERNAME}/Datasets/test_dataset/kinect/wrong_classified/gimp"

    bgs = os.listdir(f"/home/{USERNAME}/Datasets/test_dataset/kinect/wrong_classified/backgrounds")

    count = 0
    for g in gestures:
        res = os.listdir(f"{path}/bg_removed/{g}")

        for file in res:

            img = cv2.imread(f"{path}/bg_removed/{g}/{file}")

            img = cv2.fastNlMeansDenoisingColored(img, None, 1, 10, 5, 21)

            mask = cv2.inRange(img, (0, 0, 0), (5, 5, 5))

            for bg in bgs:

                img_bg = cv2.imread(f"/home/{USERNAME}/Datasets/test_dataset/kinect/wrong_classified/backgrounds/{bg}")

                img_bg[mask == 0] = img[mask == 0]

                cv2.imwrite(f"{path}/bg_changed/{g}/{bg}_{file}", img_bg)
