#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import copy
from pose_detection import PoseDetection


class HandDetectionNode:
    def __init__(self):
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)

        self.bridge = CvBridge()
        self.cv_image = None

        self.pd = PoseDetection()

        while True:
            if self.cv_image is not None:
                break

        while True:
            self.pd.cv_image = self.cv_image
            self.pd.detect_pose()
            self.pd.find_hands()

            cv2.imshow('Original Images', cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB))
            # print(self.pd.cv_image_detected_left)
            cv2.imshow('Left Hand',  cv2.cvtColor(self.pd.cv_image_detected_left, cv2.COLOR_BGR2RGB))
            cv2.imshow('Right Hand',  cv2.cvtColor(self.pd.cv_image_detected_right, cv2.COLOR_BGR2RGB))

            key = cv2.waitKey(1)

            if key == 113:
                break

    def image_callback(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")


if __name__ == '__main__':

    rospy.init_node("action_intreperter", anonymous=True)
    hd = HandDetectionNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
