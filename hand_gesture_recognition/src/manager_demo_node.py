#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from hgr.msg import HandsDetected, HandsClassified
from std_msgs.msg import Int32
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


class ManagerDemo:
    def __init__(self, thresholds, cm,**kargs) -> None:

        # Get initial data from rosparams
        print(kargs)

        image_topic = rospy.get_param("/hgr/image_topic", default="/camera/color/image_raw")
        # image_topic = rospy.get_param("/hgr/image_topic", default="/camera/rgb/image_raw")
        roi_height = rospy.get_param("/hgr/height", default=100)
        roi_width = rospy.get_param("/hgr/width", default=100)

        # Initializations for MediaPipe to detect keypoints
        self.left_hand_points = (16, 18, 20, 22)
        self.right_hand_points = (15, 17, 19, 21)

        self.bridge = CvBridge()

        rospy.Subscriber(image_topic, Image, self.image_callback)
        rospy.Subscriber("/hgr/hands", HandsDetected, self.hands_callback)
        rospy.Subscriber("/hgr/classification", HandsClassified, self.classification_callback)

        font_scale = 0.5
        gestures = ["A", "F", "L", "Y", "NONE"]

        self.cv_image = None
        self.header = None
        self.left_bounding = None
        self.right_bounding = None
        self.left_class = None
        self.right_class = None
        
        self.msg_image_buffer = {"msg": [], "timestamp": []}
        self.msg_hands_buffer = {"msg": [], "timestamp": []}
        self.msg_class_buffer = {"msg": [], "timestamp": []}

        self.bridge = CvBridge()

        cv2.namedWindow("Manager Demo", cv2.WINDOW_NORMAL)

        print("Waiting!!")
        while True:
            if len(self.msg_image_buffer["msg"]) > 0 :
                break

        while not rospy.is_shutdown():
            # print(len(self.msg_image_buffer["timestamp"]))
            st = time.time()

            msg_img, msg_hands, msg_class = self.choose_lastest_match(self.msg_image_buffer, self.msg_hands_buffer, self.msg_class_buffer)
            
            if msg_hands is not None:
                left_bounding = msg_hands.left_bounding_box
                right_bounding = msg_hands.right_bounding_box
            else:
                left_bounding = None
                right_bounding = None

            img = copy.deepcopy(cv2.cvtColor(self.bridge.imgmsg_to_cv2(msg_img, "rgb8"), cv2.COLOR_BGR2RGB))

            label_left = ""
            label_right = ""

            if msg_class is not None:

                label_left = f"Label: {msg_class.hand_left.data} - Confid: {msg_class.confid_left.data}%"

                label_right = f"Label: {msg_class.hand_right.data} - Confid: {msg_class.confid_right.data}%"

            if left_bounding is not None:
                box_left = left_bounding

                box_left_tl =  (box_left[0].data, box_left[1].data)
                box_left_br =  (box_left[2].data, box_left[3].data)

                cv2.rectangle(img, box_left_tl, box_left_br, (0, 255, 0), 2)

                if box_left_tl[1] < 30 and box_left_tl[1] > 0:
                    cv2.putText(img, label_left, (box_left_tl[0], box_left_br[1]+25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
                else:   
                    cv2.putText(img, label_left, (box_left_tl[0], box_left_tl[1]-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            
            if right_bounding is not None:

                box_right = right_bounding

                box_right_tl =  (box_right[0].data, box_right[1].data)
                box_right_br =  (box_right[2].data, box_right[3].data)

                cv2.rectangle(img, box_right_tl, box_right_br, (0, 0, 255), 2)

                if box_right_tl[1] < 30 and box_right_tl[1] > 0:
                    cv2.putText(img, label_right, ((box_right_tl[0], box_right_br[1]+25)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
                else:   
                    cv2.putText(img, label_right, (box_right_tl[0], box_right_tl[1]-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)


            cv2.imshow("Manager Demo", img)

            key = cv2.waitKey(1)

            # print(f"VISUALIZATION Running at {round(1 / (time.time() - st), 2)} FPS")

            if key == ord("q"):
                break

        cv2.destroyAllWindows()


    def image_callback(self, msg):
        # self.cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        # self.image_header = msg.header

        if len(self.msg_image_buffer["msg"]) >= 30:
             self.msg_image_buffer["msg"].pop(0)
             self.msg_image_buffer["timestamp"].pop(0)

        self.msg_image_buffer["msg"].append(msg)
        self.msg_image_buffer["timestamp"].append(msg.header.stamp.to_sec())

    def hands_callback(self, msg):
        # self.left_bounding = msg.left_bounding_box
        # self.right_bounding = msg.right_bounding_box

        if len(self.msg_hands_buffer["msg"]) >= 10:
             self.msg_hands_buffer["msg"].pop(0)
             self.msg_hands_buffer["timestamp"].pop(0)

        self.msg_hands_buffer["msg"].append(msg)
        self.msg_hands_buffer["timestamp"].append(msg.header.stamp.to_sec())
        

    def classification_callback(self, msg):
        # self.left_class = msg.status[0].values
        # self.right_class = msg.status[1].values


        if len(self.msg_class_buffer["msg"]) >= 10:
             self.msg_class_buffer["msg"].pop(0)
             self.msg_class_buffer["timestamp"].pop(0)

        self.msg_class_buffer["msg"].append(msg)
        self.msg_class_buffer["timestamp"].append(msg.header.stamp.to_sec())

    def choose_lastest_match(self, buffer1, buffer2, buffer3):
        list_time1 = buffer1["timestamp"]
        list_time2 = buffer2["timestamp"]
        list_time3 = buffer3["timestamp"]

        latest_timestamp = None
        for timestamp in list_time1:
            if timestamp in list_time2 and timestamp in list_time3:
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp

        if latest_timestamp is not None:
            # print(f'The latest matching timestamp is {latest_timestamp}')

            idx1 = buffer1["timestamp"].index(latest_timestamp)
            idx2 = buffer2["timestamp"].index(latest_timestamp)
            idx3 = buffer3["timestamp"].index(latest_timestamp)

            # print(idx3)

            return buffer1["msg"][idx1], buffer2["msg"][idx2], buffer3["msg"][idx3]
            
        else:
            # print('There are no matching timestamps in all three lists')
            return buffer1["msg"][-1], None, None




if __name__=="__main__":

    rospy.init_node("hand_gesture_recognition", anonymous=True)

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

    hd = ManagerDemo(thresholds, cm, **data)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
