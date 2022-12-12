#!/usr/bin/env python3
import copy

import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import keras
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
        font = 0.5
        gestures = ["A", "F", "L", "Y"]
        # gestures = ["G1", "G2", "G5", "G6"]

        self.model = keras.models.load_model(path + "myModel")

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

            im_array = np.asarray([left_frame, right_frame])

            prediction = self.model.predict(x=im_array, verbose=0)
            # print(prediction)

            left_frame = cv2.putText(left_frame, gestures[np.argmax(prediction[0])], org, cv2.FONT_HERSHEY_SIMPLEX,
                                     font, (0, 0, 255), thickness, cv2.LINE_AA)

            right_frame = cv2.putText(right_frame, gestures[np.argmax(prediction[1])], org, cv2.FONT_HERSHEY_SIMPLEX,
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

    hc = HandClassificationNode(path=f"{ROOT_DIR}/hand_classification/network/Resnet50/")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
