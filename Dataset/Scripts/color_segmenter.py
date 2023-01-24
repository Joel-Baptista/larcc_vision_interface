#!/usr/bin/python3
import copy
import os
import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------
# Authors: Beatriz Borges, Joel Baptista, José Cozinheiro e Tiago Fonte
# Course: PSR
# Class: Aula 7
# Date: 17 Nov. 2021
# Description: Avaliação 2 (PSR Augmented Reality Paint) - Color Segmentation File
# ----------------------------------------------------------------------------------------------------------------------

# Importing Packages
import cv2
import numpy as np
import json
from colorama import Back, Fore, Style
from skimage.color import rgb2ycbcr, gray2rgb, rgb2yiq
from sklearn.mixture import GaussianMixture

from vision_config.vision_definitions import USERNAME

# <=================================================  Global Variables  ===============================================>

global minimumb, maximumb, minimumg, maximumg, minimumr, maximumr

minimumb = 0
maximumb = 255
minimumg = 0
maximumg = 255
minimumr = 0
maximumr = 255
bias = 127

# Define trackbars' functions
def ontrackbarminb(minb):
    global minimumb
    minimumb = minb


def ontrackbarmaxb(maxb):
    global maximumb
    maximumb = maxb


def ontrackbarming(ming):
    global minimumg
    minimumg = ming - bias


def ontrackbarmaxg(maxg):
    global maximumg
    maximumg = maxg - bias


def ontrackbarminr(minr):
    global minimumr
    minimumr = minr - bias


def ontrackbarmaxr(maxr):
    global maximumr
    maximumr = maxr - bias


def get_largest_item(mask, d=2, e=1):

    # pad = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    # pad[pad.shape[0] - 1, :] = 0
    # h, w = pad.shape
    # mask = np.zeros([h + 2, w + 2], np.uint8)
    # img_floodfill = cv2.floodFill(pad, mask, (0, 0), 0, (5), (0), flags=8)[1]
    # mask = img_floodfill[1:h - 1, 1:w - 1]

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_mask = np.zeros(mask.shape)

    all_areas = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)

    largest_item = max(contours, key=cv2.contourArea)

    kernel = np.ones((3, 3), np.uint8)

    largest_mask = cv2.drawContours(largest_mask, [largest_item], -1, 255, thickness=-1)
    largest_mask = cv2.dilate(largest_mask, kernel, iterations=d)
    largest_mask = cv2.erode(largest_mask, kernel, iterations=e)
    # largest_mask = cv2.fillPoly(largest_mask, pts=largest_item, color=255)
    # largest_mask = cv2.floodFill(largest_mask, largest_item, color=255)

    return largest_mask


def main():

    # <================================================  INITIALIZATION  ==============================================>

    print(Fore.RED + "\nPSR " + Style.RESET_ALL +
          'Augmented Reality Paint, Beatriz Borges, Joel Baptista, José Cozinheiro, Tiago Fonte, November 2021\n')

    capture = cv2.VideoCapture(0)

    # Display Initial Relevant Info
    if capture.isOpened() is True:
        print('\n' + Back.GREEN + 'Starting video' + Back.RESET)
        print(Fore.RED + 'Press q to exit without saving the threshold' + Fore.RESET)
        print('\n' + Fore.CYAN + 'Press w to exit and save color limits to file' + Fore.RESET)

    else:
        print('\n' + Fore.RED + '!Error acessing Camera!' + Fore.RESET)
        print('Hint: Are you using it in another app?')

    # Create Window (600 x 600) to display Normal Image
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original', 600, 600)

    # Create Window to display Segmented Image
    window_name = 'Segmented'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # <==================================================  TRACKBARS  ================================================>

    # Implement a Set of Trackbars in Order to User be Able to Control Values (0 - 255) of Binarization Threshold
    cv2.createTrackbar('min B', window_name, 0, 255, ontrackbarminb)
    cv2.createTrackbar('max B', window_name, 0, 255, ontrackbarmaxb)
    cv2.createTrackbar('min G', window_name, 0, 255, ontrackbarming)
    cv2.createTrackbar('max G', window_name, 0, 255, ontrackbarmaxg)
    cv2.createTrackbar('min R', window_name, 0, 255, ontrackbarminr)
    cv2.createTrackbar('max R', window_name, 0, 255, ontrackbarmaxr)

    # Set Maximum Trackbars to a Default Beginning Position of 255
    cv2.setTrackbarPos('max B', window_name, 255)
    cv2.setTrackbarPos('max G', window_name, 255)
    cv2.setTrackbarPos('max R', window_name, 255)

    # Trackbars Init
    ontrackbarminb(0)
    ontrackbarmaxb(255)
    ontrackbarming(0)
    ontrackbarmaxg(255)
    ontrackbarminr(0)
    ontrackbarmaxr(255)

    # <==========================================  COLOR SEGMENTATION RESULTS  ========================================>

    df = pd.read_csv(f"/home/{USERNAME}/Datasets/test_dataset/gmm/Skin_NonSkin.txt", index_col=0)
    df.columns = ['B', 'G', 'R', 'skin']
    df['Cb'] = np.round(128 - .168736 * df.R - .331364 * df.G +
                        .5 * df.B).astype(int)
    # df['Cr'] = np.round(128 + .5 * df.R - .418688 * df.G -
    #                     .081312 * df.B).astype(int)
    df['I'] = np.round(.596 * df.R - .275 * df.G -
                       .321 * df.B).astype(int)
    df.drop(['B', 'G', 'R'], axis=1, inplace=True)
    k = 4
    print(df)
    skin_data = df[df.skin == 1].drop(['skin'], axis=1).to_numpy()
    not_skin_data = df[df.skin == 2].drop(['skin'], axis=1).to_numpy()

    skin_gmm = GaussianMixture(n_components=k, covariance_type='full').fit(skin_data)
    not_skin_gmm = GaussianMixture(n_components=k, covariance_type='full').fit(not_skin_data)

    gestures = ["A", "F", "L", "Y"]

    count = 0
    for g in gestures:

        res = os.listdir(f"/home/{USERNAME}/Datasets/test_dataset/kinect/detected/wrong/{g}")

        num_list = []
        for file in res:
            if "bg" in file:
                continue

            num = int(''.join(filter(lambda i: i.isdigit(), file)))
            num_list.append(num)

        list1, list2 = zip(*sorted(zip(num_list, res)))

        for file in list2:
            if "bg" in file:
                continue

            count += 1

            frame_original = cv2.imread(f"/home/{USERNAME}/Datasets/test_dataset/kinect/detected/wrong/{g}/{file}")

            cv2.imshow('Original', frame_original)  # Display Image from the Camera

            frame = cv2.fastNlMeansDenoisingColored(frame_original, None, 1, 10, 5, 21)
            frame_ycbcr = 127 * rgb2yiq(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            while True:
                # Achieve Ranges for each Value of RGB Channels
                ranges = {'limits': {'B': {'min': minimumb, 'max': maximumb},
                                     'G': {'min': minimumg, 'max': maximumg},
                                     'R': {'min': minimumr, 'max': maximumr}}}

                # Convert Ranges Dictionary into Numpy Arrays
                mins = np.array([0, ranges['limits']['G']['min'], ranges['limits']['R']['min']])
                maxs = np.array([255, ranges['limits']['G']['max'], ranges['limits']['R']['max']])
                # print(mins)
                # print(maxs)
                # print(np.max(frame))
                # print(np.min(frame))
                # Create Mask for Color Segmentation

                mask = cv2.inRange(frame_ycbcr, mins, maxs)
                mask_hsv = cv2.inRange(frame_hsv, (minimumb, 0, 0), (maximumb, 255, 255))

                mask2 = get_largest_item(mask)
                mask_hsv2 = get_largest_item(mask_hsv)

                proc_image1 = rgb2ycbcr(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))[:, :, 1]
                proc_image2 = 255 * rgb2yiq(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))[:, :, 1]
                proc_image = np.reshape(cv2.merge([proc_image1, proc_image2]), (-1, 2))

                skin_score = skin_gmm.score_samples(proc_image)
                not_skin_score = not_skin_gmm.score_samples(proc_image)
                result = skin_score > not_skin_score

                mask_gmm = result.reshape(frame.shape[0], frame.shape[1]).astype(np.uint8) * 255
                mask_gmm = get_largest_item(mask_gmm, 3, 1)

                mask2 = cv2.bitwise_and(mask_hsv2, mask2)
                mask2 = cv2.bitwise_and(mask2, mask_gmm)

                segmented_image = copy.deepcopy(frame_original)
                segmented_image[mask2 == 0] = [0, 0, 0]

                # Apply and show Created Mask in Color Segmentation Window (600 x 600)
                cv2.namedWindow('Segmented', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Segmented', 600, 600)
                cv2.namedWindow('Largest Segmented', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Largest Segmented', 600, 600)

                cv2.imshow('Segmented', mask2)
                cv2.imshow('Mask HSV', mask_hsv2)
                cv2.imshow('Mask GMM', mask_gmm)
                cv2.imshow('Largest Segmented', segmented_image)
                # <======================================  SAVE / NOT SAVE PROGRESS  =========================================>

                key = cv2.waitKey(5)  # Wait a Key to stop the Program

                if key == 13:
                    break

                if key == ord("s"):
                    cv2.imwrite(f"/home/{USERNAME}/Datasets/test_dataset/kinect/detected/wrong/{g}/bg/{file}", segmented_image)
                    print(f"{file} save!")
                    break

                # Use "q" or "Q" (Quit) Key to End the Program without saving the JSON File
                if key == 113 or key == 81:
                    print('\nProgram ending without saving progress')
                    break

                # Use "w" or "W" (Write) Key to End the Program saving and writing the JSON File
                elif key == 119 or key == 87:
                    file_name = 'limits_lowLight.json'
                    with open(file_name, 'w') as file_handle:
                        print('\nWriting color limits to file ' + file_name)
                        print(ranges)
                        json.dump(ranges, file_handle)
                    break

            if key == ord('q'):
                break

        if key == ord('q'):
            break
        # <=================================================  TERMINATING  ===============================================>

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
