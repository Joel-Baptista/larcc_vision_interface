#!/usr/bin/env python3
import rospy
from hgr.msg import detected_hands
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
import cv2
import numpy as np
from cv_bridge import CvBridge
import copy
import time
import mediapipe as mp
from yaml.loader import SafeLoader
from vision_config.vision_definitions import ROOT_DIR
from hand_gesture_recognition.utils.hgr_utils import find_hands



class HandDetectionNode:
    def __init__(self):

        image_topic = rospy.get_param("/hgr/image_topic", default="/camera/color/image_raw")
        # image_topic = rospy.get_param("/hgr/image_topic", default="/camera/rgb/image_raw")
        roi_height = rospy.get_param("/hgr/height", default=100)
        roi_width = rospy.get_param("/hgr/width", default=100)  

        fps = rospy.get_param("/hgr/FPS/detection")

        # Initializations for MediaPipe to detect keypoints
        # self.left_hand_points = (16, 18, 20, 22)
        # self.right_hand_points = (15, 17, 19, 21)

        self.bridge = CvBridge()

        # self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_drawing_styles = mp.solutions.drawing_styles
        # self.mp_pose = mp.solutions.pose
        # self.pose = self.mp_pose.Pose(static_image_mode=False,
        #                               model_complexity=2,
        #                               enable_segmentation=False,
        #                               min_detection_confidence=0.7)

        mp_data = {}
        mp_data["left_hand_points"] = (16, 18, 20, 22)
        mp_data["right_hand_points"] = (15, 17, 19, 21)
        mp_data["mp_drawing"] = mp.solutions.drawing_utils
        mp_data["mp_drawing_styles"] = mp.solutions.drawing_styles
        mp_data["mp_pose"] = mp.solutions.pose
        mp_data["pose"] = mp_data["mp_pose"].Pose(static_image_mode=False,
                                                model_complexity=2,
                                                enable_segmentation=False,
                                                min_detection_confidence=0.7)


        rospy.Subscriber(image_topic, Image, self.image_callback)
        pub_hands = rospy.Publisher("/hgr/hands", detected_hands, queue_size=10)

        self.cv_image = None
        self.image_header = None
        self.msg = None
        self.bridge = CvBridge()


        print("Waiting!!")
        while True:
            if self.cv_image is not None:
                break
                
        try:
            while not rospy.is_shutdown():
                st = time.time()
                image = copy.deepcopy(self.cv_image)
                header = copy.deepcopy(self.image_header)

                # left_bounding, right_bounding, hand_right, hand_left, keypoints_image = self.find_hands(image, x_lim=int(roi_width / 2), y_lim=int(roi_height / 2))

                left_bounding, right_bounding, hand_right, hand_left, keypoints_image, mp_data["pose"] = find_hands(
                copy.deepcopy(self.cv_image), mp_data, x_lim=int(roi_width / 2), y_lim=int(roi_height / 2))


                left_b = [Int32(i) for i in left_bounding]
                right_b = [Int32(i) for i in right_bounding]

                hands = detected_hands()
                hands.header = header
                hands.left_bounding_box = list(left_b)
                hands.right_bounding_box = list(right_b)

                if hand_left is not None:
                    # left_hand = cv2.resize(left_hand, (200, 200), interpolation=cv2.INTER_CUBIC)
                    hands.hand_left = self.bridge.cv2_to_imgmsg(hand_left, "rgb8")

                if hand_right is not None:
                    # right_hand = cv2.resize(right_hand, (200, 200), interpolation=cv2.INTER_CUBIC)
                    hands.hand_right = self.bridge.cv2_to_imgmsg(hand_right, "rgb8")
               

                while True:

                    if time.time() - st > 1/fps:
                        break
                    
                    time.sleep(1 / (fps * 1000))

                pub_hands.publish(hands)
                # print(f"DETECTION Running at {round(1 / (time.time() - st), 2)} FPS")

        except KeyboardInterrupt:
            print("Shutting down")
            cv2.destroyAllWindows()
        
    def image_callback(self, msg):
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.image_header = msg.header
        self.msg = msg

    # def find_hands(self, input_image, x_lim, y_lim):

    #     hand_left_bounding_box = [0, 0, 0, 0]
    #     hand_right_bounding_box = [0, 0, 0, 0]

    #     h, w, _ = input_image.shape
    #     image = copy.deepcopy(input_image)

    #     results = self.pose.process(image)

    #     annotated_image = image.copy()

    #     self.mp_drawing.draw_landmarks(
    #         annotated_image,
    #         results.pose_landmarks,
    #         self.mp_pose.POSE_CONNECTIONS,
    #         landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())


    #     x_left_points = []
    #     x_right_points = []
    #     y_left_points = []
    #     y_right_points = []

    #     h, w, _ = image.shape

    #     hand_left = None
    #     hand_right = None

    #     if results.pose_landmarks:
    #         for id_landmark, landmark in enumerate(results.pose_landmarks.landmark):
    #             if id_landmark in self.left_hand_points:
    #                 x_left_points.append(landmark.x)
    #                 y_left_points.append(landmark.y)

    #             if id_landmark in self.right_hand_points:
    #                 x_right_points.append(landmark.x)
    #                 y_right_points.append(landmark.y)

    #         l_c = [int(np.mean(x_left_points) * w), int(np.mean(y_left_points) * h)]
    #         r_c = [int(np.mean(x_right_points) * w), int(np.mean(y_right_points) * h)]


    #         if l_c[0] < x_lim:
    #             l_c[0] = x_lim
    #         if l_c[1] < y_lim:
    #             l_c[1] = y_lim
    #         if r_c[0] < x_lim:
    #             r_c[0] = x_lim
    #         if r_c[1] < y_lim:
    #             r_c[1] = y_lim


    #         hand_left_bounding_box = [l_c[0]-x_lim, l_c[1]-y_lim, l_c[1]+y_lim, l_c[0]+x_lim]
    #         hand_right_bounding_box = [l_c[0]-x_lim, r_c[1]-y_lim, l_c[0]+x_lim, l_c[1]+y_lim]
            
    #         hand_left = input_image[l_c[1]-y_lim:l_c[1]+y_lim,
    #                                                     l_c[0]-x_lim:l_c[0]+x_lim]

    #         hand_right = input_image[r_c[1]-y_lim:r_c[1]+y_lim,
    #                                                      r_c[0]-x_lim:r_c[0]+x_lim]

    #         left_start_point = (l_c[0]-x_lim, l_c[1]-y_lim)
    #         left_end_point = (l_c[0]+x_lim, l_c[1]+y_lim)

    #         right_start_point = (r_c[0]-x_lim, r_c[1]-y_lim)
    #         right_end_point = (r_c[0]+x_lim, r_c[1]+y_lim)

    #         cv2.rectangle(annotated_image, left_start_point, left_end_point, (255, 0, 0), 2)
    #         cv2.rectangle(annotated_image, right_start_point, right_end_point, (255, 0, 0), 2)

    #     if np.array(hand_left).shape != (2*x_lim, 2*y_lim, 3):
    #         hand_left = None

    #     if np.array(hand_right).shape != (2*x_lim, 2*y_lim, 3):
    #         hand_right = None

    #     return hand_left_bounding_box, hand_right_bounding_box, hand_right, hand_left, annotated_image


if __name__ == '__main__':

    rospy.init_node("hand_detection", anonymous=True)

    hd = HandDetectionNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
