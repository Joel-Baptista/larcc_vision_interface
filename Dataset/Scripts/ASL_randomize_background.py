#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt

from vision_config.vision_definitions import USERNAME, ROOT_DIR
import os
import json
import random
import numpy as np
import copy
import pandas as pd
from sklearn.mixture import GaussianMixture
from skimage.color import rgb2ycbcr, gray2rgb, rgb2yiq


def get_largest_item(mask):

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_mask = np.zeros(mask.shape)

    all_areas = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)

    largest_item = max(contours, key=cv2.contourArea)

    kernel = np.ones((3, 3), np.uint8)

    largest_mask = cv2.drawContours(largest_mask, [largest_item], -1, 255, thickness=-1)
    largest_mask = cv2.dilate(largest_mask, kernel)
    # largest_mask = cv2.fillPoly(largest_mask, pts=largest_item, color=255)
    # largest_mask = cv2.floodFill(largest_mask, largest_item, color=255)

    return largest_mask


if __name__ == '__main__':

    # Load GMM Model

    df = pd.read_csv(f'/home/{USERNAME}/Datasets/ASL/Skin_NonSkin.txt', index_col=0)
    df['Cb'] = np.round(128 - .168736 * df.R - .331364 * df.G +
                        .5 * df.B).astype(int)
    # df['Cr'] = np.round(128 + .5 * df.R - .418688 * df.G -
    #                     .081312 * df.B).astype(int)
    df['I'] = np.round(.596 * df.R - .275 * df.G -
                       .321 * df.B).astype(int)
    df.drop(['B', 'G', 'R'], axis=1, inplace=True)

    k = 7

    skin_data = df[df.skin == 1].drop(['skin'], axis=1).to_numpy()
    not_skin_data = df[df.skin == 2].drop(['skin'], axis=1).to_numpy()

    skin_gmm = GaussianMixture(n_components=k, covariance_type='full').fit(skin_data)
    not_skin_gmm = GaussianMixture(n_components=k, covariance_type='full').fit(not_skin_data)
    colors = ['navy', 'turquoise', 'darkorange', 'gold']

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

        res = os.listdir(f"/home/{USERNAME}/Datasets/ASL/train/{g}")

        num_list = []
        for file in res:
            num = int(''.join(filter(lambda i: i.isdigit(), file)))
            num_list.append(num)

        list1, list2 = zip(*sorted(zip(num_list, res)))

        for file in list2:
            count += 1
            if (count / total_images) * 100 >= last_percentage:
                print(f"{last_percentage}% of images analysed")
                last_percentage += 10

            r = random.sample(range(0, 1500), 3)

            r_scale = [x / 1000 for x in r]

            img = cv2.imread(f"/home/{USERNAME}/Datasets/{dataset}/train/{g}/{file}")
            #
            # histg = cv2.calcHist([img], [0], None, [256], [0, 256])
            #
            # plt.plot(histg)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            img_use = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img_use = cv2.bilateralFilter(img_use, 10, 20, 20)

            proc_image1 = rgb2ycbcr(img_use)[:, :, 1]
            proc_image2 = 255 * rgb2yiq(img_use)[:, :, 1]
            proc_image = np.reshape(cv2.merge([proc_image1, proc_image2]), (-1, 2))
            skin_score = skin_gmm.score_samples(proc_image)
            not_skin_score = not_skin_gmm.score_samples(proc_image)
            mask_gmm = skin_score > not_skin_score

            mask_gmm = (255 * mask_gmm.reshape(img.shape[0], img.shape[1]).astype(int))

            mask_hsv = cv2.inRange(img_hsv, (7, 0, 0), (255, 20, 255))
            mask1 = cv2.inRange(img, (lower_limit, lower_limit, lower_limit), (255, 255, 255))

            result = cv2.bitwise_and(img, img, mask=mask_hsv)
            result = cv2.multiply(result, (r_scale[0], r_scale[1], r_scale[2], 1))

            img_result = copy.deepcopy(img)
            mask_inv = cv2.bitwise_not(mask_hsv)
            img_result = cv2.bitwise_or(img_result, img_result, mask=mask_inv)

            img_result = cv2.bitwise_or(result, img_result)

            # cv2.imwrite(f"/home/{USERNAME}/Datasets/{dataset}/background_augmented/{g}/{file}", img_result)

            # result = np.bitwise_and(cv2.cvtColor(255 * result.astype(np.uint8), cv2.COLOR_GRAY2BGR), img)

            mask2 = get_largest_item(mask_gmm.astype(np.uint8))
            together = cv2.bitwise_and(mask_inv, mask2.astype(np.uint8))

            segmented_image = copy.deepcopy(img)
            segmented_image[together == 0] = [0, 0, 0]

            cv2.imshow("Mask GMM", segmented_image)
            cv2.imshow("Together", together)

            cv2.imshow("test", img)
            cv2.imshow("mask", mask_inv)
            cv2.imshow("result", img_result)

            key = cv2.waitKey()

            if key == ord("s"):
                cv2.imwrite(f"/home/{USERNAME}/Datasets/ASL/train_bg_removed/{g}/{file}", segmented_image)
                print(f"{file} save!")

            if key == ord('q'):
                break
        if key == ord('q'):
            break
