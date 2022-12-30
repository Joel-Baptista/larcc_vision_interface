#!/usr/bin/env python3
import copy
import numpy as np
import cv2
from cv_bridge import CvBridge
from vision_config.vision_definitions import ROOT_DIR
import mediapipe as mp


class PoseDetection:
    def __init__(self, static_image_mode=False):

        self.left_hand_points = (16, 18, 20, 22)
        self.right_hand_points = (15, 17, 19, 21)

        self.x_lim_l = 50
        self.y_lim_l = 50

        self.x_lim_r = 50
        self.y_lim_r = 50

        self.bridge = CvBridge()
        self.cv_image = None
        self.cv_image_detected = None
        self.cv_image_detected_left = None
        self.cv_image_detected_right = None
        self.results = None

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=static_image_mode,
                                      model_complexity=2,
                                      enable_segmentation=False,
                                      min_detection_confidence=0.7)

    def detect_pose(self):

        h, w, _ = self.cv_image.shape
        image = copy.deepcopy(self.cv_image)
        # self.results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.results = self.pose.process(image)

        annotated_image = image.copy()
        # condition = np.stack((self.results.segmentation_mask,) * 3, axis=-1) > 0.1

        self.cv_image_detected = annotated_image

        self.mp_drawing.draw_landmarks(
            self.cv_image_detected,
            self.results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

    def find_hands(self, x_lim=112, y_lim=112):

        x_left_points = []
        x_right_points = []
        y_left_points = []
        y_right_points = []

        h, w, _ = self.cv_image.shape

        # self.cv_image_detected_left = np.zeros((self.x_lim_l * 2, self.y_lim_l * 2, 3), np.uint8)
        # self.cv_image_detected_right = np.zeros((self.x_lim_r * 2, self.y_lim_r * 2, 3), np.uint8)
        #
        self.cv_image_detected_left = None
        self.cv_image_detected_right = None

        if self.results.pose_landmarks:
            for id_landmark, landmark in enumerate(self.results.pose_landmarks.landmark):
                if id_landmark in self.left_hand_points:
                    x_left_points.append(landmark.x)
                    y_left_points.append(landmark.y)

                if id_landmark in self.right_hand_points:
                    x_right_points.append(landmark.x)
                    y_right_points.append(landmark.y)

            l_c = [int(np.mean(x_left_points) * w), int(np.mean(y_left_points) * h)]
            r_c = [int(np.mean(x_right_points) * w), int(np.mean(y_right_points) * h)]

            # THIS COMMENTED CODE TRIES TO MAKE AN ADAPTATIVE WINDOW. FIRST WE WILL TRY FIXED WINDOW
            #
            # l_max = [int(max(x_left_points) * w), int(max(y_left_points) * h)]
            # l_min = [int(min(x_left_points) * w), int(min(y_left_points) * h)]
            #
            # r_max = [int(max(x_right_points) * w), int(max(y_right_points) * h)]
            # r_min = [int(min(x_right_points) * w), int(min(y_right_points) * h)]
            #
            # if len(x_left_points) == 4:
            #     self.x_lim_l = int(2.5 * max([l_max[0] - l_min[0], l_max[1] - l_min[1]]))
            #     self.y_lim_l = self.x_lim_l
            #
            # if len(x_right_points) == 4:
            #     self.x_lim_r = int(2.5 * max([r_max[0] - r_min[0], r_max[1] - r_min[1]]))
            #     self.y_lim_r = self.x_lim_r
            #
            # if l_c[0] < self.x_lim_l:
            #     l_c[0] = self.x_lim_l
            # if l_c[1] < self.y_lim_l:
            #     l_c[1] = self.y_lim_l
            # if r_c[0] < self.x_lim_r:
            #     r_c[0] = self.x_lim_r
            # if r_c[1] < self.y_lim_r:
            #     r_c[1] = self.y_lim_r
            #
            # self.cv_image_detected_left = self.cv_image[l_c[1]-self.y_lim_l:l_c[1]+self.y_lim_l,
            #                                             l_c[0]-self.x_lim_l:l_c[0]+self.x_lim_l]
            #
            # self.cv_image_detected_right = self.cv_image[r_c[1]-self.y_lim_r:r_c[1]+self.y_lim_r,
            #                                              r_c[0]-self.x_lim_r:r_c[0]+self.x_lim_r]
            #
            # left_start_point = (l_c[0]-self.x_lim_l, l_c[1]-self.y_lim_l)
            # left_end_point = (l_c[0]+self.x_lim_l, l_c[1]+self.y_lim_l)
            #
            # right_start_point = (r_c[0]-self.x_lim_r, r_c[1]-self.y_lim_r)
            # right_end_point = (r_c[0]+self.x_lim_r, r_c[1]+self.y_lim_r)

            if l_c[0] < x_lim:
                l_c[0] = x_lim
            if l_c[1] < y_lim:
                l_c[1] = y_lim
            if r_c[0] < x_lim:
                r_c[0] = x_lim
            if r_c[1] < y_lim:
                r_c[1] = y_lim

            self.cv_image_detected_left = self.cv_image[l_c[1]-y_lim:l_c[1]+y_lim,
                                                        l_c[0]-x_lim:l_c[0]+x_lim]

            self.cv_image_detected_right = self.cv_image[r_c[1]-y_lim:r_c[1]+y_lim,
                                                         r_c[0]-x_lim:r_c[0]+x_lim]

            left_start_point = (l_c[0]-x_lim, l_c[1]-y_lim)
            left_end_point = (l_c[0]+x_lim, l_c[1]+y_lim)

            right_start_point = (r_c[0]-x_lim, r_c[1]-y_lim)
            right_end_point = (r_c[0]+x_lim, r_c[1]+y_lim)

            cv2.rectangle(self.cv_image_detected, left_start_point, left_end_point, (255, 0, 0), 2)
            cv2.rectangle(self.cv_image_detected, right_start_point, right_end_point, (255, 0, 0), 2)

        if np.array(self.cv_image_detected_left).shape != (2*x_lim, 2*y_lim, 3):
            self.cv_image_detected_left = None

        if np.array(self.cv_image_detected_right).shape != (2*x_lim, 2*y_lim, 3):
            self.cv_image_detected_right = None

        # if len(self.cv_image_detected_left) == 0:
        #     # self.cv_image_detected_left = np.zeros((100, 100, 3), np.uint8)
        #     self.cv_image_detected_left = None
        # if len(self.cv_image_detected_right) == 0:
        #     # self.cv_image_detected_right = np.zeros((100, 100, 3), np.uint8)
        #     self.cv_image_detected_right = None
