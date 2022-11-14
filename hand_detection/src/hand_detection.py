#!/usr/bin/env python3
import cv2
import numpy as np
from cv_bridge import CvBridge
import mediapipe as mp
import copy


class HandDetection:
    def __init__(self):

        self.bridge = CvBridge()
        self.cv_image = None
        self.cv_image_detected = None

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=2)
        self.mpDraw = mp.solutions.drawing_utils

    def detect_hands(self):
        cx_list = [[], []]
        cy_list = [[], []]

        frame = copy.deepcopy(self.cv_image)
        results = self.hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_id, handLms in enumerate(results.multi_hand_landmarks):  # working with each hand
                print(hand_id)
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # print(cx_list)
                    if hand_id < 2:
                        cx_list[hand_id].append(cx)
                        cy_list[hand_id].append(cy)

                    if id == 15:
                        cv2.circle(frame, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

                self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

        self.cv_image_detected = frame

            # self.cv_image_detected = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # cx_mean = [300, 300]
            # cy_mean = [300, 300]
            #
            # for i in range(0, len(cx_list)):
            #     if len(cx_list[i]) > 0 and len(cx_list[i]) > 0:
            #         cx_mean[i] = int(np.mean(cx_list[i]))
            #         cy_mean[i] = int(np.mean(cy_list[i]))

            # # hand_frame = frameBGR[cy_mean-50:cy_mean+50, cx_mean-50:cx_mean+50]
            # first_hand_frame = cv2.cvtColor(self.cv_image[cy_mean[0] - 50:cy_mean[0] + 50, cx_mean[0] - 50:cx_mean[0] + 50],
            #                                 cv2.COLOR_RGB2BGR)
            #
            # second_hand_frame = cv2.cvtColor(self.cv_image[cy_mean[1] - 50:cy_mean[1] + 50, cx_mean[1] - 50:cx_mean[1] + 50],
            #                                  cv2.COLOR_RGB2BGR)
            #
            # cv2.imshow("first_hand_image", first_hand_frame)
            # cv2.imshow("second_hand_image", second_hand_frame)
            #
            # cv2.imshow("Output", frameBGR)

