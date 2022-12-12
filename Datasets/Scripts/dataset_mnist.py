#!/usr/bin/env python3
import pandas as pd
from vision_config.vision_definitions import ROOT_DIR
import numpy as np
import cv2


def count_samples(samples: pd.DataFrame, file: str):
    count = 0
    label_list = []
    count_list = []

    for item in samples["label"]:
        count += 1

        if item in label_list:
            count_list[label_list.index(item)] += 1
        else:
            label_list.append(item)
            count_list.append(0)

    label_list, count_list = zip(*sorted(zip(label_list, count_list)))

    print(label_list)
    print(count_list)

    data_frame = {"Label": label_list,
                  "Size": count_list}

    df = pd.DataFrame(data_frame)
    df.to_csv(ROOT_DIR + f"/Datasets/MNIST/{file}_sample_size.csv")


file_name = "sign_mnist_train"

# Read the csv file
data = pd.read_csv(ROOT_DIR + f'/Datasets/MNIST/{file_name}.csv')

# Print it out if you want

