#!/usr/bin/env python3


import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_config.vision_definitions import ROOT_DIR, USERNAME
import os
import json
import time
import numpy as np
from pose_detection import PoseDetection
import copy


class DatasetCapture:
    def __init__(self, fps=30, save_path=f"/home/{USERNAME}/Datasets/sliding_window"):
        # rospy.Subscriber("/camera/rgb/image_raw", Image, self.get_image_callback)
        rospy.Subscriber("/camera/color/image_raw", Image, self.get_image_callback) # Astra

        self.frame = None
        self.bridge = CvBridge()
        self.width = None
        self.height = None

        self.pd = PoseDetection()

        with open(f'{ROOT_DIR}/Dataset/configs/larcc_dataset_config.json') as f:
            config = json.load(f)

        # config[dataset]["gestures"] = ["None"]

        # for dataset in config:
        #     gestures = config[dataset]["gestures"]
        #     for g in gestures:
        #         if not os.path.exists(f"{save_path}/{g}"):
        #             os.mkdir(f"{save_path}/{g}")

        print("Waiting!!")
        while True:
            if self.frame is not None:
                break

        st = time.time()
        record = False
        buffer_left = []
        buffer_right = []
        cv2.namedWindow("Video feed", cv2.WINDOW_NORMAL)

        while True:

            self.pd.cv_image = copy.deepcopy(self.frame)
            self.pd.detect_pose()
            self.pd.find_hands(x_lim=50, y_lim=50)

            cv2.imshow('Video feed', self.pd.cv_image_detected)
            key = cv2.waitKey(1)

            if not record and len(buffer_right) > 0:
                res = os.listdir(f"{save_path}/right")
                shift = len(res)
                print("Start saving right images")
                for i, image in enumerate(buffer_right):
                    cv2.imwrite(f"{save_path}/right/{i+shift}.png", image)

                buffer_right = []
                print("Right Images Saved")


            if not record and len(buffer_left) > 0:
                res = os.listdir(f"{save_path}/left")
                shift = len(res)
                print("Start saving left images")
                for i, image in enumerate(buffer_left):
                    cv2.imwrite(f"{save_path}/left/{i+shift}.png", image)

                buffer_left = []
                print("Left Images Saved")


            if time.time() - st >= 1/fps and record:
                st = time.time()
                buffer_left.append(np.array(self.pd.cv_image_detected_left))
                buffer_right.append(np.array(self.pd.cv_image_detected_right))

            if key == ord('q'):
                break

            if key == 13:
                record = not record
                if record:
                    print("Start recording in ...")
                    # for j in range(1, 4):
                    #     print(j)
                    #     time.sleep(1)
                    # print("GO")
                else:
                    print("End recording")

        cv2.destroyAllWindows()

    def get_image_callback(self, msg):
        self.frame = cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg, "rgb8"), cv2.COLOR_BGR2RGB)

        if self.height is None:
            self.height = msg.height
            print(f"Height: {self.height}")

        if self.width is None:
            self.width = msg.width
            print(f"Width: {self.width}")


if __name__ == '__main__':
    rospy.init_node("dataset_creation", anonymous=True)
    dc = DatasetCapture()
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print("Shutting down")

    cv2.destroyAllWindows()
