#!/usr/bin/env python3
import cv2
from vision_config.vision_definitions import USERNAME, ROOT_DIR
import os
import json
import random
import numpy as np
import copy

if __name__ == '__main__':

    dataset = "ASL"
    dataset_path = f"/home/{USERNAME}/Datasets/{dataset}/train"

    with open(f'{ROOT_DIR}/Dataset/configs/larcc_dataset_config.json') as f:
        config = json.load(f)

    if not os.path.exists(f"/home/{USERNAME}/Datasets/{dataset}/background_augmented"):
        os.mkdir(f"/home/{USERNAME}/Datasets/{dataset}/background_augmented")

    for gesture in config[dataset]["gestures"]:
        if os.path.exists(f"/home/{USERNAME}/Datasets/{dataset}/background_augmented/{gesture}"):
            os.rmdir(f"/home/{USERNAME}/Datasets/{dataset}/background_augmented/{gesture}")
            os.mkdir(f"/home/{USERNAME}/Datasets/{dataset}/background_augmented/{gesture}")
        else:
            os.mkdir(f"/home/{USERNAME}/Datasets/{dataset}/background_augmented/{gesture}")

    total_images = 0
    for g in config[dataset]["gestures"]:
        res = os.listdir(f"/home/{USERNAME}/Datasets/{dataset}/train/{g}")
        total_images += len(res)

    lower_limit = 200
    last_percentage = 0

    count = 0
    for g in config[dataset]["gestures"]:

        res = os.listdir(f"/home/{USERNAME}/Datasets/{dataset}/train/{g}")

        for file in res:
            count += 1
            if (count / total_images) * 100 >= last_percentage:
                print(f"{last_percentage}% of images analysed")
                last_percentage += 10

            r = random.sample(range(0, 1500), 3)

            r_scale = [x / 1000 for x in r]

            img = cv2.imread(f"/home/{USERNAME}/Datasets/{dataset}/train/{g}/{file}")

            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            mask_hsv = cv2.inRange(img_hsv, (15, 0, 0), (255, 30, 255))
            mask = cv2.inRange(img, (lower_limit, lower_limit, lower_limit), (255, 255, 255))

            result = cv2.bitwise_and(img, img, mask=mask_hsv)
            result = cv2.multiply(result, (r_scale[0], r_scale[1], r_scale[2], 1))

            img_result = copy.deepcopy(img)
            mask_inv = cv2.bitwise_not(mask_hsv)
            img_result = cv2.bitwise_or(img_result, img_result, mask=mask_inv)

            img_result = cv2.bitwise_or(result, img_result)

            cv2.imwrite(f"/home/{USERNAME}/Datasets/{dataset}/background_augmented/{g}/{file}", img_result)

            # cv2.imshow("test", img)
            # cv2.imshow("mask", mask_hsv)
            # cv2.imshow("result", img_result)
            #
            # key = cv2.waitKey(100)
            #
            # if key == ord('q'):
            #     break
