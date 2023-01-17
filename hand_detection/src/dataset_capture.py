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
    def __init__(self, fps=5, save_path=f"/home/{USERNAME}/Datasets/Larcc_dataset/astra"):
        # rospy.Subscriber("/camera/rgb/image_raw", Image, self.get_image_callback)
        rospy.Subscriber("/camera/color/image_raw", Image, self.get_image_callback) # Astra

        self.frame = None
        self.bridge = CvBridge()
        self.width = None
        self.height = None

        self.pd = PoseDetection()

        with open(f'{ROOT_DIR}/Dataset/configs/larcc_dataset_config.json') as f:
            config = json.load(f)

        for dataset in config:
            gestures = config[dataset]["gestures"]
            for g in gestures:
                if not os.path.exists(f"{save_path}/{g}"):
                    os.mkdir(f"{save_path}/{g}")

        print("Waiting!!")
        while True:
            if self.frame is not None:
                break

        st = time.time()
        record = False
        buffer = []

        while True:

            self.pd.cv_image = copy.deepcopy(self.frame)
            self.pd.detect_pose()
            self.pd.find_hands(x_lim=100, y_lim=100)

            cv2.imshow('Video feed', self.pd.cv_image_detected)
            key = cv2.waitKey(1)

            if not record and len(buffer) > 0:
                classification = input("Input image label: ")

                print("Start saving images")
                for i, image in enumerate(buffer):
                    if i <= len(buffer) - fps:
                        cv2.imwrite(f"{save_path}/{str(classification).upper()}/image{i}.png", image)

                buffer = []
                print("Images Saved")

            if time.time() - st >= 1/fps and record:
                st = time.time()
                buffer.append(np.array(self.frame))

            if key == ord('q'):
                break

            if key == 13:
                record = not record
                if record:
                    print("Start recording in ...")
                    for j in range(1, 4):
                        print(j)
                        time.sleep(1)
                    print("GO")
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
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
