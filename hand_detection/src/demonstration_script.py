#!/usr/bin/env python3
from pose_detection import PoseDetection
import os
import cv2


if __name__ == '__main__':
    dataset_path = "/home/joelbaptista/Desktop/DEMO"

    input_folder = "Original"
    output_folder = "NN Input"

    pd = PoseDetection()

    if os.path.exists(f"{dataset_path}/{input_folder}"):

        res = os.listdir(f"{dataset_path}/{input_folder}")

        for file in res:

            img = cv2.imread(f"{dataset_path}/{input_folder}/{file}")

            pd.cv_image = img
            pd.detect_pose()
            pd.find_hands(x_lim=50, y_lim=50)

            if (pd.cv_image_detected_right is not None) and (pd.cv_image_detected_left is not None):
                cv2.imwrite(f"{dataset_path}/{output_folder}/left_{file}", pd.cv_image_detected_left)
                cv2.imwrite(f"{dataset_path}/{output_folder}/right_{file}", pd.cv_image_detected_right)

            # cv2.imshow("test", pd.cv_image_detected)
            # key = cv2.waitKey()
            #
            # if key == ord('q'):
            #     break

