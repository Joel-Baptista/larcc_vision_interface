#!/usr/bin/env python3
import cv2
import numpy as np
from vision_config.vision_definitions import USERNAME
import os


class BackgroundBlur:
    def __int__(self):

        self.img = None

    def background_segementation(self):

        # frame = cv2.GaussianBlur(self.img, (7, 7), 0)

        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        hsv_mask1 = cv2.inRange(hsv, (0, 30, 0), (15, 255, 255))

        kernel = np.ones((3, 3), np.float32)

        # hsv_mask1 = cv2.erode(hsv_mask1, kernel, iterations=1)

        image_contours = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

        image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

        contours = cv2.findContours(hsv_mask1, 1, 1)[0]

        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area:
                cv2.fillPoly(image_binary, pts=[contour], color=(255, 255, 255))
                cv2.drawContours(image_contours, contour, -1, (255, 255, 255), 1)

        #
        # for contour in contours:
        #     cv2.drawContours(image_binary, contour,
        #                      -1, (255, 255, 255), -1)
        # #
        # image_binary = cv2.erode(image_binary, kernel, iterations=1)
        image_binary = cv2.dilate(image_binary, kernel, iterations=1)

        # ret, min_sat = cv2.threshold(s, 30, 250, cv2.THRESH_BINARY)
        # ret, max_hue = cv2.threshold(h, 15, 255, cv2.THRESH_BINARY_INV)
        # min_sat = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)
        # max_hue = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

        # final = cv2.bitwise_and(min_sat, max_hue)

        masked_image = cv2.bitwise_and(self.img, self.img, mask=image_binary)

        print("final")
        # print(sum(sum(final > 1)))
        print("hsv_mask1")
        print(sum(sum(hsv_mask1 > 1)))
        cv2.imshow("Original", self.img)
        cv2.imshow("Image Binary", image_binary)
        cv2.imshow("HSV mask", hsv_mask1)
        cv2.imshow("Masked_image", masked_image)

        key = cv2.waitKey()

        if key == ord('q'):
            exit()


if __name__ == '__main__':

    # dataset_path = f"/home/{USERNAME}/Datasets/ASL/test"
    dataset_path = f"/home/{USERNAME}/Datasets/Larcc_dataset/test_ASL"

    gestures = ["A", "F", "L", "Y"]

    bb = BackgroundBlur()

    for g in gestures:

        res = os.listdir(f"{dataset_path}/{g}")

        for file in res:

            img = cv2.imread(f"{dataset_path}/{g}/{file}")

            bb.img = img

            bb.background_segementation()
