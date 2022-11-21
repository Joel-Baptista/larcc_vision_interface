#!/usr/bin/env python3
import copy
import os
import time
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from vision_config.vision_definitions import ROOT_DIR
from pose_detection import PoseDetection
import numpy as np


class GetVideo:
    def __init__(self, save_path):
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)

        self.bridge = CvBridge()
        self.cv_image = None
        self.height = None
        self.width = None
        self.timestamp = []

        pd = PoseDetection()

        while len(self.timestamp) < 30:
            pass

        delays = [x - self.timestamp[i - 1] for i, x in enumerate(self.timestamp)][1:]

        print("delays")
        print(delays)

        fps = int(1 / np.mean(delays))

        print("fps")
        print(fps)

        print("self.width")
        print(self.width)
        print("self.height")
        print(self.height)

        self.writer = cv2.VideoWriter(save_path + '/test.avi',
                                      cv2.VideoWriter_fourcc('I', '4', '2', '0'),
                                      20,
                                      (self.width, self.height))

        while True:
            pd.cv_image = copy.deepcopy(self.cv_image)
            pd.detect_pose()
            # pd.find_hands()

            # frame = copy.deepcopy(pd.cv_image_detected)
            frame = cv2.cvtColor(pd.cv_image_detected, cv2.COLOR_BGR2RGB)

            self.writer.write(frame)

            cv2.imshow('Original Images', frame)

            key = cv2.waitKey(1)

            if key == 113:
                break

    def image_callback(self, msg):
        self.width = msg.width
        self.height = msg.height
        self.timestamp.append(time.time())

        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")


if __name__ == '__main__':
    rospy.init_node("get_kinect_video", anonymous=True)

    count = 0
    while count > 0:
        print(f"In {count}")
        count -= 1
        time.sleep(1)

    gv = GetVideo(save_path="/home/joelbaptista/Desktop")

    gv.writer.release()
    cv2.destroyAllWindows()
