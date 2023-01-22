#!/usr/bin/env python3
import os
import numpy as np
import cv2
from vision_config.vision_definitions import USERNAME
import pandas as pd
from random import randint

if __name__ == '__main__':

    df = pd.read_csv(f'/home/{USERNAME}/Datasets/Skin_NonSkin.txt', header=None, delim_whitespace=True)
    df.columns = ['B', 'G', 'R', 'skin']

    df = df.drop(df[df.skin == 2].index)
    print(df)

    bg_dataset_path = f"/home/{USERNAME}/Datasets/ASL/gmm_background/"
    fg_dataset_path = f"/home/{USERNAME}/Datasets/ASL/gmm_foreground/"

    res_bg = os.listdir(bg_dataset_path)
    res_fg = os.listdir(fg_dataset_path)

    dic_bg = {"R": [],
              "G": [],
              "B": [],
              "skin": []}

    R = []
    G = []
    B = []
    no_skin = []
    for image_name in res_bg:

        img = cv2.imread(bg_dataset_path + image_name)

        # cv2.imshow("test", img)
        # cv2.waitKey()

        r = img[:, :, 2].flatten()
        g = img[:, :, 1].flatten()
        b = img[:, :, 0].flatten()
        # r = np.reshape(img[:, :, 2], (1, -1))
        # g = np.reshape(img[:, :, 1], (1, -1))
        # b = np.reshape(img[:, :, 0], (1, -1))

        for i in range(0, len(r)):
            #
            # dic_row = {"R": r[i],
            #            "G": g[i],
            #            "B": b[i],
            #            "skin": 2}

            if randint(0, 100) < 15:
                dic_bg["R"].append(r[i])
                dic_bg["G"].append(g[i])
                dic_bg["B"].append(b[i])
                dic_bg["skin"].append(2)

    print(np.array(dic_bg["R"]).shape)
    for image_name in res_fg:

        img = cv2.imread(fg_dataset_path + image_name)

        # cv2.imshow("test", img)
        # cv2.waitKey()

        r = img[:, :, 2].flatten()
        g = img[:, :, 1].flatten()
        b = img[:, :, 0].flatten()
        # r = np.reshape(img[:, :, 2], (1, -1))
        # g = np.reshape(img[:, :, 1], (1, -1))
        # b = np.reshape(img[:, :, 0], (1, -1))

        for i in range(0, len(r)):
            #
            # dic_row = {"R": r[i],
            #            "G": g[i],
            #            "B": b[i],
            #            "skin": 2}

            dic_bg["R"].append(r[i])
            dic_bg["G"].append(g[i])
            dic_bg["B"].append(b[i])
            dic_bg["skin"].append(1)

    df = pd.DataFrame(dic_bg)

    df.to_csv(f"/home/{USERNAME}/Datasets/ASL/Skin_NonSkin.txt")

    print(df)

