#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge
import mediapipe as mp


class HandDetection:
    def __init__(self):
        rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)

        self.bridge = CvBridge()
        self.cv_image = None

        while True:
            if self.cv_image is not None:
                break

        mpHands = mp.solutions.hands
        hands = mpHands.Hands()
        mpDraw = mp.solutions.drawing_utils

        while True:
            cx_list = []
            cy_list = []

            frame = self.cv_image
            results = hands.process(frame)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:  # working with each hand
                    cx_list = []
                    cy_list = []
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)

                        cx_list.append(cx)
                        cy_list.append(cy)

                        if id==20:
                            cv2.circle(frame, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

                    mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            frameBGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cx_mean = 300
            cy_mean = 300

            print(len(cx_list))
            print(len(cy_list))
            if len(cx_list) > 0 and len(cx_list) > 0:

                cx_mean = int(np.mean(cx_list))
                cy_mean = int(np.mean(cy_list))

            hand_frame = frameBGR[cy_mean-50:cy_mean+50, cx_mean-50:cx_mean+50]

            cv2.imshow("hand_image", hand_frame)

            cv2.imshow("Output", frameBGR)
            key = cv2.waitKey(10)

            if key == 113:
                break

    def image_callback(self, msg):

        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")



if __name__ == '__main__':

    rospy.init_node("action_intreperter", anonymous=True)
    hd = HandDetection()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()
