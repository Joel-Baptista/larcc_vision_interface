#!/usr/bin/python3
import copy
import os

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
    minimumb = minb - bias


def ontrackbarmaxb(maxb):
    global maximumb
    maximumb = maxb - bias


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


def get_largest_item(mask):

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_mask = np.zeros(mask.shape)

    all_areas = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)

    largest_item = max(contours, key=cv2.contourArea)

    kernel = np.ones((3, 3), np.uint8)

    largest_mask = cv2.drawContours(largest_mask, [largest_item], -1, 255, thickness=-1)
    largest_mask = cv2.dilate(largest_mask, kernel)
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
            frame = 127 * rgb2yiq(cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB))
            # frame = cv2.cvtColor(frame_original, cv2.COLOR_BGR2HSV)

            while True:
                # Achieve Ranges for each Value of RGB Channels
                ranges = {'limits': {'B': {'min': minimumb, 'max': maximumb},
                                     'G': {'min': minimumg, 'max': maximumg},
                                     'R': {'min': minimumr, 'max': maximumr}}}

                # Convert Ranges Dictionary into Numpy Arrays
                mins = np.array([ranges['limits']['B']['min'], ranges['limits']['G']['min'], ranges['limits']['R']['min']])
                maxs = np.array([ranges['limits']['B']['max'], ranges['limits']['G']['max'], ranges['limits']['R']['max']])
                # print(mins)
                # print(maxs)
                # print(np.max(frame))
                # print(np.min(frame))
                # Create Mask for Color Segmentation
                mask = cv2.inRange(frame, mins, maxs)

                mask2 = get_largest_item(mask)

                segmented_image = copy.deepcopy(frame_original)
                segmented_image[mask2 == 0] = [0, 0, 0]

                # Apply and show Created Mask in Color Segmentation Window (600 x 600)
                cv2.imshow('Segmented', mask)
                cv2.imshow('Largest Segmented', segmented_image)
                cv2.namedWindow('Segmented', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Segmented', 600, 600)

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
