#!/usr/bin/env python3
import cv2
from vision_config.vision_definitions import USERNAME
from hand_detection.src.pose_detection import PoseDetection
import os
import copy


if __name__ == '__main__':

    # dataset_path = f"/home/{USERNAME}/Datasets/ASL/test"
    dataset_path = f"/home/{USERNAME}/Datasets/Larcc_dataset/larcc_test_1"
    saving_path = f"/home/{USERNAME}/Datasets/ASL"
    gestures = ["A", "F", "L", "Y"]
    # gestures = ["L", "Y"]
    pd = PoseDetection()

    for g in gestures:

        res = os.listdir(f"{dataset_path}/{g}")

        for file in res:

            img = cv2.imread(f"{dataset_path}/{g}/{file}")

            pd.cv_image = copy.deepcopy(img)
            pd.detect_pose()
            pd.find_hands(x_lim=75, y_lim=75)

            if pd.cv_image_detected_right is not None:
                cv2.imshow("Image Right", pd.cv_image_detected_right)

                cv2.imwrite(f"{saving_path}/larcc_addition/{g}/{file}", pd.cv_image_detected_right)

            if pd.cv_image_detected_left is not None:
                cv2.imshow("Image Left", cv2.flip(pd.cv_image_detected_left, 1))

                cv2.imwrite(f"{dataset_path}/larcc_addition/{g}/{file}", cv2.flip(pd.cv_image_detected_left, 1))

            cv2.imshow("Original", pd.cv_image_detected)
            key = cv2.waitKey(1)

            if key == ord('q'):
                exit()




