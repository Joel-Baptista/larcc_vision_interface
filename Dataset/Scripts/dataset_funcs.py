import copy

import cv2
import numpy as np
from skimage.color import rgb2ycbcr, rgb2yiq
from sklearn.mixture import GaussianMixture

from hand_detection.src.pose_detection import PoseDetection
import pandas as pd

from vision_config.vision_definitions import USERNAME


def preprocessing(images, process):

    if "gmm remove background" in process:
        print("Removing Background")
        processed_images = gmm_remove_background(images, process[1])
    else:
        print("No Processing")
        processed_images = images

    return processed_images


def gmm_remove_background(images, dataset_path):

    bg_rm_images = []

    df = pd.read_csv(dataset_path, header=None, delim_whitespace=True)
    df.columns = ['B', 'G', 'R', 'skin']
    df['Cb'] = np.round(128 - .168736 * df.R - .331364 * df.G +
                        .5 * df.B).astype(int)
    # df['Cr'] = np.round(128 + .5 * df.R - .418688 * df.G -
    #                     .081312 * df.B).astype(int)
    df['I'] = np.round(.596 * df.R - .275 * df.G -
                       .321 * df.B).astype(int)
    df.drop(['B', 'G', 'R'], axis=1, inplace=True)
    k = 4

    skin_data = df[df.skin == 1].drop(['skin'], axis=1).to_numpy()
    not_skin_data = df[df.skin == 2].drop(['skin'], axis=1).to_numpy()

    skin_gmm = GaussianMixture(n_components=k, covariance_type='full').fit(skin_data)
    not_skin_gmm = GaussianMixture(n_components=k, covariance_type='full').fit(not_skin_data)

    for image in images:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        proc_image1 = rgb2ycbcr(image_rgb)[:, :, 1]
        proc_image2 = 255 * rgb2yiq(image_rgb)[:, :, 1]
        proc_image = np.reshape(cv2.merge([proc_image1, proc_image2]), (-1, 2))

        skin_score = skin_gmm.score_samples(proc_image)
        not_skin_score = not_skin_gmm.score_samples(proc_image)
        result = skin_score > not_skin_score

        result = result.reshape(image.shape[0], image.shape[1]).astype(np.uint8) * 255

        # Remove objects from top, left and right
        pad = cv2.copyMakeBorder(result, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        pad[pad.shape[0]-1, :] = 0
        h, w = pad.shape
        mask = np.zeros([h + 2, w + 2], np.uint8)
        img_floodfill = cv2.floodFill(pad, mask, (0, 0), 0, (5), (0), flags=8)[1]
        result = img_floodfill[1:h - 1, 1:w - 1]

        contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_mask = np.zeros(result.shape)

        all_areas = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            all_areas.append(area)

        if len(contours) > 0:
            largest_item = max(contours, key=cv2.contourArea)

            kernel = np.ones((3, 3), np.uint8)

            largest_mask = cv2.drawContours(largest_mask, [largest_item], -1, 255, thickness=-1)
            largest_mask = cv2.dilate(largest_mask, kernel)

            segmented_image = copy.deepcopy(image)
            segmented_image[largest_mask == 0] = [0, 0, 0]

            bg_rm_images.append(segmented_image)

    return bg_rm_images
