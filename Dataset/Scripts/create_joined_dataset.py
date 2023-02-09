#!/usr/bin/env python3
import cv2
from vision_config.vision_definitions import USERNAME
from hand_detection.src.pose_detection import PoseDetection
import os
import copy
from shutil import copyfile
import Augmentor


if __name__ == '__main__':

    dataset1 = f"{os.getenv('HOME')}/Datasets/ASL/kinect/train"
    dataset2 = f"{os.getenv('HOME')}/Datasets/ASL/kinect/kinect_daniel"
    dataset3 = f"{os.getenv('HOME')}/Datasets/ASL/kinect/kinect_lucas"
    dataset4 = f"{os.getenv('HOME')}/Datasets/ASL/kinect/kinect_manel"

    list = [dataset1, dataset2, dataset3]
    
    dataset_destination = f"{os.getenv('HOME')}/Datasets/ASL/kinect/train_joined"
    gestures = ["A", "F", "L", "Y"]

    os.mkdir(dataset_destination)

    for g in gestures:
        count = 0
        os.mkdir(os.path.join(dataset_destination, g))

        for dataset in list:
            res = os.listdir(f"{dataset}/{g}")

            for file in res:
                
                copyfile(f"{dataset}/{g}/{file}", f"{dataset_destination}/{g}/image{count}.png")
                count = count + 1
