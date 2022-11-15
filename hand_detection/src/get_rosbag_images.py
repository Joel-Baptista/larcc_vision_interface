#!/usr/bin/env python3
import os

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from vision_config.vision_definitions import ROOT_DIR


class GetRosbagImages:
    def __init__(self, path):
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)

        self.bridge = CvBridge()
        self.cv_image = None
        res = os.listdir(path)
        count = len(res)

        while self.cv_image is None:
            pass

        while True:
            cv2.imshow('Original Images', cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB))

            key = cv2.waitKey(1)

            if key == 13:
                frame = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(path + f"/image{count}.png", frame)
                print(f"Saved image{count}")
                count += 1

            if key == 113:
                break

    def image_callback(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")


if __name__ == '__main__':
    rospy.init_node("rosbag_snapshot", anonymous=True)

    path = ROOT_DIR + "/Datasets/Joel"

    gri = GetRosbagImages(path=path)
