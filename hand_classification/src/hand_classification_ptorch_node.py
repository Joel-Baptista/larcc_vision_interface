#!/usr/bin/env python3
import copy
import os
from torchvision import datasets, transforms
from PIL import Image as PIL_Image
from hand_classification.network.ptorch.networks import InceptionV3
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import torch

import time

from vision_config.vision_definitions import ROOT_DIR


class HandClassificationNode:
    def __init__(self, path=f"{ROOT_DIR}/hand_classification/network/"):
        rospy.Subscriber("/hand/left", Image, self.left_hand_callback)
        rospy.Subscriber("/hand/right", Image, self.right_hand_callback)
        self.left_hand = None
        self.right_hand = None
        self.bridge = CvBridge()

        org = (0, 90)
        thickness = 1
        font = 1
        gestures = ["A", "F", "L", "Y"]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = InceptionV3(4, 0.0001, unfreeze_layers=list(np.arange(13, 20)), class_features=2048, device=device,
                            con_features=64)
        model.name = "InceptionV3_multi_loss"

        trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{model.name}/{model.name}.pth',
                                     map_location=torch.device(device))
        model.load_state_dict(trained_weights)

        model.eval()

        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.25, 0.25, 0.25])

        data_transforms = transforms.Compose([
            transforms.Resize(299),
            # transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        print("Waiting!!")
        while True:
            if self.left_hand is not None and self.right_hand is not None:
                break

        while True:
            # cv2.imshow('Classifier - Left Hand', cv2.cvtColor(self.left_hand, cv2.COLOR_BGR2RGB))
            # cv2.imshow('Classifier - Right Hand', cv2.cvtColor(self.right_hand, cv2.COLtransformsOR_BGR2RGB))

            left_frame = copy.deepcopy(cv2.cvtColor(self.left_hand, cv2.COLOR_BGR2RGB))
            right_frame = copy.deepcopy(cv2.cvtColor(self.right_hand, cv2.COLOR_BGR2RGB))

            if left_frame.shape != (100, 100, 3) or right_frame.shape != (100, 100, 3):
                continue

            left_img = PIL_Image.fromarray(left_frame)
            right_img = PIL_Image.fromarray(cv2.flip(right_frame, 1))

            # left_img = PIL_Image.fromarray(cv2.flip(left_frame, 1))
            # right_img = PIL_Image.fromarray(right_frame)

            left_tensor = data_transforms(left_img)
            right_tensor = data_transforms(right_img)

            inputs = torch.stack([left_tensor, right_tensor], 0)

            with torch.no_grad():
                outputs, _ = model(inputs)
                _, preds = torch.max(outputs, 1)

            print(preds)
            left_frame = cv2.putText(left_frame, gestures[preds[0]], org, cv2.FONT_HERSHEY_SIMPLEX,
                                     font, (0, 0, 255), thickness, cv2.LINE_AA)

            right_frame = cv2.putText(right_frame, gestures[preds[1]], org, cv2.FONT_HERSHEY_SIMPLEX,
                                      font, (0, 0, 255), thickness, cv2.LINE_AA)

            cv2.imshow('Left Hand Classifier', left_frame)
            cv2.imshow('Right Hand Classifier', right_frame)

            # print(f"Prediction time: {round(time.time() - st, 2)} seconds")

            key = cv2.waitKey(1)

            if key == 113:
                break

        cv2.destroyAllWindows()

    def left_hand_callback(self, msg):
        self.left_hand = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def right_hand_callback(self, msg):
        self.right_hand = self.bridge.imgmsg_to_cv2(msg, "rgb8")


if __name__ == '__main__':

    rospy.init_node("hand_classification", anonymous=True)

    hc = HandClassificationNode(path=f"{ROOT_DIR}/hand_classification/models/InceptionV3_larcc_contrastive/")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
