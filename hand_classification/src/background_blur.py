#!/usr/bin/env python3
import cv2
import numpy as np
from vision_config.vision_definitions import USERNAME
import os
from hand_detection.src.pose_detection import PoseDetection
import copy
import matplotlib.pyplot as plt


class BackgroundBlur:
    def __init__(self):
        self.final_images = []
        self.final_image = None
        self.img = None
        self.final_mask = None
        self.blurred_stats = None

    def background_segementation(self):

        self.final_images = []

        blurred_images = []
        # self.blurred_stats = [(7, 7, 0),
        #                       (7, 7, 10),
        #                       (7, 7, 20),
        #                       (7, 7, 30),
        #                       (7, 7, 40)]
        self.blurred_stats = [(3, 3, 0),
                              (7, 7, 0),
                              (15, 15, 0),
                              (23, 23, 0),
                              (33, 33, 0)]

        for blurred_stat in self.blurred_stats:
            blurred_image = cv2.GaussianBlur(self.img, (blurred_stat[0], blurred_stat[1]), blurred_stat[2])
            blurred_images.append(blurred_image)

        images_to_show = []

        cv2.imshow("Original", self.img)

        images_to_show.append(self.img)

        images_to_show.append(blurred_images[0])

        noiseless_image_colored = cv2.fastNlMeansDenoisingColored(self.img, None, 3, 21, 7, 21)

        cv2.imshow("Noiseless image", noiseless_image_colored)

        images_to_show.append(noiseless_image_colored)

        hsv = cv2.cvtColor(noiseless_image_colored, cv2.COLOR_BGR2HSV)

        hsv_mask1 = cv2.inRange(hsv, (0, 30, 0), (15, 255, 255))

        cv2.imshow("Preprocessed BW", hsv_mask1)

        images_to_show.append(hsv_mask1)

        image_binary = np.zeros((self.img.shape[0], self.img.shape[1], 1), np.uint8)

        # add 1 pixel white border all around
        pad = cv2.copyMakeBorder(hsv_mask1, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        h, w = pad.shape

        # create zeros mask 2 pixels larger in each dimension
        mask = np.zeros([h + 2, w + 2], np.uint8)

        # floodfill outer white border with black
        img_floodfill = cv2.floodFill(pad, mask, (0, 0), 0, (5), (0), flags=8)[1]

        # remove border
        hsv_mask1 = img_floodfill[1:h - 1, 1:w - 1]

        cv2.imshow("Borders Removed BW", hsv_mask1)

        images_to_show.append(hsv_mask1)

        # contours = cv2.findContours(hsv_mask1, 1, 1)[0]
        contours = cv2.findContours(hsv_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        if len(contours) != 0:
            # print(contours)
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]

            cv2.drawContours(image_binary, cnt, -1, 255, -1)

        mask_binary = np.zeros([image_binary.shape[0] + 2, image_binary.shape[1] + 2], np.uint8)

        image_binary = cv2.floodFill(image_binary, mask_binary, (0, 0), 255)[1]
        image_binary = cv2.bitwise_not(image_binary)

        kernel = np.ones((3, 3))

        image_binary = cv2.dilate(image_binary, kernel, iterations=5)
        image_binary = cv2.erode(image_binary, kernel, iterations=3)

        self.final_mask = image_binary
        cv2.imshow("Final BW", image_binary)

        images_to_show.append(image_binary)

        masked_image = cv2.bitwise_and(self.img, self.img, mask=image_binary)

        cv2.imshow("Masked image", masked_image)
        images_to_show.append(masked_image)

        image_binary_inv = cv2.bitwise_not(image_binary)

        count = 0
        for blurred_image in blurred_images:
            self.final_image = cv2.bitwise_and(blurred_image, blurred_image, mask=image_binary_inv)
            self.final_image = cv2.add(self.final_image, masked_image)
            count += 1
            self.final_images.append(self.final_image)
            # cv2.imshow(f"Blurred Image {count}", self.final_image)

        images_to_show.append(self.final_image)

        cv2.imshow("Blurred Backgourd", self.final_image)
        # self.show_images(images_to_show)

        # for i in range(0, len(blurred_images)):
        #
        #     cv2.imshow(f"{self.blurred_stats[i]}", self.final_images[i])
        #
        # key = cv2.waitKey()
        #
        # if key == ord('q'):
        #     exit()

    def show_images(self, images) -> None:
        n: int = len(images)
        f = plt.figure()
        for i in range(n):
            # Debug, plot figure
            f.add_subplot(2, 4, i + 1)
            plt.imshow(images[i])

        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        plt.show(block=True)


if __name__ == '__main__':

    dataset_path = f"/home/{USERNAME}/Datasets/ASL/train"
    # dataset_path = f"/home/{USERNAME}/Datasets/Larcc_dataset/home_test"

    gestures = ["A", "F", "L", "Y"]
    # gestures = ["L", "Y"]

    bb = BackgroundBlur()
    pd = PoseDetection()

    for g in gestures:

        res = os.listdir(f"{dataset_path}/{g}")

        for file in res:

            img = cv2.imread(f"{dataset_path}/{g}/{file}")

            pd.cv_image = copy.deepcopy(img)
            pd.detect_pose()
            pd.find_hands(x_lim=75, y_lim=75)

            # if pd.cv_image_detected_right is not None:
            #     cv2.imshow("Image", pd.cv_image_detected_right)
            #
            #     cv2.imwrite(f"{dataset_path}/detected/{g}/{file}", pd.cv_image_detected_right)
            #
            # cv2.imshow("Original", pd.cv_image_detected)
            # key = cv2.waitKey()
            #
            # if key == ord('q'):
            #     exit()
            #
            bb.img = copy.deepcopy(img)
            # bb.img = copy.deepcopy(pd.cv_image_detected_left)

            bb.background_segementation()

            cv2.imshow("Original Image", bb.img)
            cv2.imshow("Final Mask", bb.final_mask)
            cv2.imshow("Final Image", bb.final_image)

            key = cv2.waitKey()

            if key == ord('y'):
                for i in range(0, len(bb.final_images)):
                    stats = bb.blurred_stats[i]
                    cv2.imwrite(f"{dataset_path}/blurred_{stats[0]}_{stats[1]}/{g}/{file}",
                                bb.final_images[i])

                print("Images saved")

            if key == ord('q'):
                exit()


