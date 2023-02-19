#!/usr/bin/env python3
import copy
import os
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
from hand_classification.network.ptorch.networks import InceptionV3
import torch
import argparse
import time
from torchvision import datasets, transforms
from PIL import Image as Im
from vision_config.vision_definitions import ROOT_DIR


class HandClassificationNode:
    def __init__(self, args, path=f"{ROOT_DIR}/hand_classification/network/"):
        rospy.Subscriber("/hand/left", Image, self.left_hand_callback)
        rospy.Subscriber("/hand/right", Image, self.right_hand_callback)
        self.left_hand = None
        self.right_hand = None
        self.bridge = CvBridge()

        org = (0, 90)
        thickness = 1
        font = 1
        gestures = ["A", "F", "L", "Y"]
        # gestures = ["G1", "G2", "G5", "G6"]
        # model = choose_model(args.model_name, device)

        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print("Training device: ", device)

        model = InceptionV3(4, 0.0001, unfreeze_layers= list(np.arange(13, 19)), device=device, con_features=16)
        model.name = args.model_name

        trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{model.name}/{model.name}.pth', map_location=torch.device(device))
        model.load_state_dict(trained_weights)


        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.25, 0.25, 0.25])

        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(299),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        print("Waiting!!")
        while True:
            if self.left_hand is not None and self.right_hand is not None:
                break

        while True:
            # cv2.imshow('Classifier - Left Hand', cv2.cvtColor(self.left_hand, cv2.COLOR_BGR2RGB))
            # cv2.imshow('Classifier - Right Hand', cv2.cvtColor(self.right_hand, cv2.COLOR_BGR2RGB))

            left_frame = copy.deepcopy(cv2.cvtColor(self.left_hand, cv2.COLOR_BGR2RGB))
            right_frame = copy.deepcopy(cv2.cvtColor(self.right_hand, cv2.COLOR_BGR2RGB))

            if left_frame.shape != (200, 200, 3) or right_frame.shape != (200, 200, 3):
                continue

            im_array = np.asarray([left_frame, cv2.flip(right_frame, 1)])

            # tensor1 = torch.Tensor(left_frame).unsqueeze(0)
            # print(tensor1.shape)
            # tensor2 = torch.Tensor(cv2.flip(right_frame, 1)).unsqueeze(0)
            # print(tensor2.shape)

            # tensor_array = torch.cat((tensor1, tensor2), 0)
            # print(tensor_array.shape)

            im1 = data_transform(left_frame)
            im2 = data_transform(cv2.flip(right_frame, 1))

            im_array_norm = torch.cat((im1.unsqueeze(0), im2.unsqueeze(0)), 0)

            with torch.no_grad():
                outputs, _ = model(im_array_norm)
                _, preds = torch.max(outputs, 1)
            
            print(preds)
            print(outputs)
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

    parser = argparse.ArgumentParser(
                prog = 'Pytorch Test',
                description = 'It test a Pytorch model')

    parser.add_argument('-d', '--device', type=str, default="cuda:1", help='Decive used for testing')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('-m', '--model_name', type=str, default="InceptionV3", help='Model name')
    parser.add_argument('-t', '--test_dataset', type=str, default="kinect_test", help='Test dataset name')

    args = parser.parse_args()

    rospy.init_node("hand_classification", anonymous=True)

    hc = HandClassificationNode(args, path=f'{os.getenv("HOME")}/Datasets/ASL/kinect/{args.model_name}')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
