#!/usr/bin/env python3
import cv2
from vision_config.vision_definitions import USERNAME
from hand_detection.src.pose_detection import PoseDetection
import os
import copy
import Augmentor


if __name__ == '__main__':

    dataset1 = f"/home/{USERNAME}/Datasets/ASL/larcc_addition"
    dataset2 = f"/home/{USERNAME}/Datasets/ASL/train"

    gestures = ["A", "F", "L", "Y"]

    for g in gestures:

        res = os.listdir(f"{dataset1}/{g}")

        num_samples = int(10 * len(res))

        # aug_data[f"user{subject}"][gesture] = num_samples

        p = Augmentor.Pipeline(source_directory=f"{dataset1}/{g}",
                               output_directory=f"/home/{USERNAME}/Datasets/ASL/joined_dataset/{g}")

        p.rotate(probability=1,
                 max_left_rotation=15,
                 max_right_rotation=15)

        p.zoom(probability=1,
               min_factor=0.9,
               max_factor=1.1)

        p.zoom_random(probability=1,
                      percentage_area=0.8,
                      randomise_percentage_area=False)

        # p.random_brightness(probability=0.25,
        #                     min_factor=0.9,
        #                     max_factor=1.1)
        #
        # p.random_color(probability=0.25,
        #                min_factor=0.2,
        #                max_factor=1.0)
        #
        # p.random_contrast(probability=0.25,
        #                   min_factor=0.2,
        #                   max_factor=1.0)

        # p.random_erasing(probability=0.25,
        #                  rectangle_area=0.4)

        p.sample(num_samples)

    # for g in gestures:
    #
    #     res = os.listdir(f"{dataset2}/{g}")
    #
    #     for file in res:
    #
    #         img = cv2.imread(f"{dataset2}/{g}/{file}")
    #         cv2.imwrite(f"/home/{USERNAME}/Datasets/ASL/joined_dataset/{g}/{file}", img)
