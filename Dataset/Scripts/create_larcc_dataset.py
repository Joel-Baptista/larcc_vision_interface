#!/usr/bin/env python3
import cv2
from vision_config.vision_definitions import USERNAME
from hand_detection.src.pose_detection import PoseDetection
import os
import copy


if __name__ == '__main__':

    # dataset_path = f"/home/{USERNAME}/Datasets/ASL/test"
    dataset_path = f"/home/{USERNAME}/Datasets/Larcc_dataset/kinect_test"
    saving_path = f"/home/{USERNAME}/Datasets/ASL/test_kinect"
    gestures = ["A", "F", "L", "Y"]
    # gestures = ["L", "Y"]
    pd = PoseDetection()

    for g in gestures:

        res = os.listdir(f"{dataset_path}/{g}")

        num_list = []
        for file in res:
            if "bg" in file:
                continue

            num = int(''.join(filter(lambda i: i.isdigit(), file)))
            num_list.append(num)

        list1, list2 = zip(*sorted(zip(num_list, res)))

        count = 0
        for file in list2:

            img = cv2.imread(f"{dataset_path}/{g}/{file}")

            pd.cv_image = copy.deepcopy(img)
            pd.detect_pose()
            pd.find_hands(x_lim=50, y_lim=50)

            if pd.cv_image_detected_right is not None:
                cv2.imshow("Image Right", pd.cv_image_detected_right)
                count += 1
                cv2.imwrite(f"{saving_path}/{g}/{count}_{file}", pd.cv_image_detected_right)

            if pd.cv_image_detected_left is not None:
                cv2.imshow("Image Left", cv2.flip(pd.cv_image_detected_left, 1))
                count += 1
                cv2.imwrite(f"{saving_path}/{g}/{count}_{file}", cv2.flip(pd.cv_image_detected_left, 1))

            cv2.imshow("Original", pd.cv_image_detected)
            key = cv2.waitKey()

            if key == ord('q'):
                exit()




