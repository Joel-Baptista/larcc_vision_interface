#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import copy
import mediapipe as mp
import os
from hand_gesture_recognition.utils.networks import InceptionV3
import torch
import time
from torchvision import transforms
import yaml
from yaml.loader import SafeLoader
from vision_config.vision_definitions import ROOT_DIR


class HandClassificationNode:
    def __init__(self, thresholds, cm,**kargs) -> None:
       # Get initial data from rosparams
        print(kargs)

        # Initializations for MediaPipe to detect keypoints
        self.left_hand_points = (16, 18, 20, 22)
        self.right_hand_points = (15, 17, 19, 21)

        self.bridge = CvBridge()

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      model_complexity=2,
                                      enable_segmentation=False,
                                      min_detection_confidence=0.7)

        rospy.Subscriber("/hgr/hand_left", Image, self.left_hand_callback)
        rospy.Subscriber("/hgr/hand_right", Image, self.right_hand_callback)

        self.cv_image = None
        self.right_msg = None
        self.left_msg = None
        self.bridge = CvBridge()

        # Initialize variables for classification
        self.thresholds = thresholds

        buffer_left = [4] * kargs["n_frames"] # Initializes the buffer with 5 "NONE" gestures
        buffer_right = [4] * kargs["n_frames"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Training device: ", self.device)

        self.model = InceptionV3(4, 0.0001, unfreeze_layers=list(np.arange(13, 20)), class_features=2048, device=self.device,
                    con_features=kargs["con_features"])
        self.model.name = kargs["model_name"]

        trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{self.model.name}/{self.model.name}.pth', map_location=torch.device(self.device))
        self.model.load_state_dict(trained_weights)

        self.model.eval()

        self.model.to(self.device)

        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.25, 0.25, 0.25])

        self.data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        print("Waiting!!")
        while True:
            if self.cv_image is not None:
                break

        try:
            while True:

                left_frame = copy.deepcopy(cv2.cvtColor(self.left_hand, cv2.COLOR_BGR2RGB))
                right_frame = copy.deepcopy(cv2.cvtColor(self.right_hand, cv2.COLOR_BGR2RGB))

                if left_frame.shape != (100, 100, 3) or right_frame.shape != (100, 100, 3):
                    continue

                if left_frame is not None:
                    
                    pred_left, buffer_left = self.take_decision(buffer_left, left_frame, cm, flip=True)

                else:

                    pred_left = 4

                if left_frame:
                    pred_right, buffer_right = self.take_decision(buffer_right, left_frame, cm, flip=False)

                else:

                    pred_right = 4

        except KeyboardInterrupt:
            print("Shutting down")
            cv2.destroyAllWindows()

    def take_decision(self, buffer, hand, cm, flip = True):

        if flip:
            hand = cv2.flip(hand, 1)

        im_norm = self.data_transform(hand).unsqueeze(0)
        im_norm = im_norm.to(self.device)

        with torch.no_grad():   
            outputs, _ = self.model(im_norm)
            _, preds = torch.max(outputs, 1)

        if outputs[0][preds] <= self.thresholds[preds]:
            preds = 4

        buffer.pop(0)
        buffer.append(preds)

        pred = 4

        buffer_positives = [i for i in buffer if i != 4] # Removes "None" class

        # Decision heuristic 1

        # all_equal = all(element == buffer_positives[0] for element in buffer_positives) # Checks if all items are equal to each other

        # positives_ratio = len(buffer_positives) / len(buffer)

        # if all_equal and positives_ratio > 0.3:
        #     pred = buffer_positives[0]

        # Decision heuristic 2

        # if len(buffer_positives):
        #     counter = 0
        #     num = buffer_positives[0]
        
        #     for i in buffer_positives: # get most frequent classification
        #         curr_frequency = buffer_positives.count(i)
        #         if(curr_frequency> counter):
        #             counter = curr_frequency
        #             num = i

        #     most_frequent_ratio = counter / len(buffer)
        #     most_frequent_positives_ratio = counter / len(buffer_positives)

        #     if most_frequent_ratio > t_most_frequent_ratio and most_frequent_positives_ratio > t_most_frequent_positives_ratio:

        #         pred = num

        # Decision heuristic 3

        probability = []
        confidance = []

        for i in range(0, 5):

            prob = 0

            for prediction in buffer:
                # prob = prob * (cm[i][prediction] / 100 + 0.01) # Multiplicate probabilities

                prob = prob + cm[i][prediction] / (100 * len(buffer)) # Average of probabilities

            probability.append(prob)
            confidance.append(prob / (cm[i][i] / 100))

        pred = probability.index(max(probability))

        if flip:
            print("--------------------------------------------")
            print(f"Buffer: {buffer}")
            print(f"probability: {probability}")
            print(f"confidance: {confidance}")
            print(f"Prediction: {pred}")

        return pred, buffer


    def left_hand_callback(self, msg):
        self.left_hand = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.left_msg = msg

    def right_hand_callback(self, msg):
        self.right_hand = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.right_msg = msg

if __name__ == '__main__':

    rospy.init_node("hand_classification", anonymous=True)

    model_name = rospy.get_param("/hgr/model_name", default="InceptionV3")

    with open(f'{ROOT_DIR}/hand_gesture_recognition/config/model/{model_name}.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
        print(data)

    with open(f'{ROOT_DIR}/hand_gesture_recognition/config/model/thresholds.yaml') as f:
        t = yaml.load(f, Loader=SafeLoader)
        print(t)

    thresholds = t["thresholds"][data["threshold_choice"]]
    cm = t["confusion_matrix"][data["threshold_choice"]]
    print(thresholds)
    print(cm)


    hc = HandClassificationNode(thresholds, cm, **data)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()