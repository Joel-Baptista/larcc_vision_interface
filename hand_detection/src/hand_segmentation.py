#!/usr/bin/env python3
import copy

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
import seaborn as sns
from skimage.io import imread
from skimage.color import rgb2ycbcr, gray2rgb, rgb2yiq
from vision_config.vision_definitions import USERNAME


def draw_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        covariances = gmm.covariances_[n][:2, :2]
    v, w = np.linalg.eigh(covariances)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                              180 + angle, color=color)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)
    ax.set_aspect('equal', 'datalim')


def normalize_channel(img):

    img_norm = np.array(copy.deepcopy(img))

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    print(b)
    channel_sum = b+g+r
    print(channel_sum)
    img_norm[:, :, 0] = b/channel_sum * 255.0
    img_norm[:, :, 1] = g/channel_sum * 255.0
    img_norm[:, :, 2] = r/channel_sum * 255.0

    # img_norm = cv2.convertScaleAbs(img_norm)

    # for i in range(0, img.shape[0]):
    #     for j in range(0, img.shape[1]):
    #
    #         b = img[i, j, 0]
    #         g = img[i, j, 1]
    #         r = img[i, j, 2]
    #
    #         img_norm[i, j, 0] = round(255 * (b / (b + g + r)), 0)
    #         img_norm[i, j, 1] = round(255 * (g / (b + g + r)), 0)
    #         img_norm[i, j, 2] = round(255 * (r / (b + g + r)), 0)

    return img_norm


if __name__ == '__main__':


    # Loads dataset
    df = pd.read_csv(f'/home/{USERNAME}/Datasets/Skin_NonSkin.txt', header=None, delim_whitespace=True)
    df.columns = ['B', 'G', 'R', 'skin']

    # Plot the distribution of the RGB values for skin and nonskin examples (1 is skin, 2 is no skin)
    g = sns.catplot(data=pd.melt(df, id_vars='skin'),
                    x='variable', y='value', hue='variable', col='skin',
                    kind='box', palette=sns.color_palette("hls", 3)[::-1])

    # Obtain cb and cr values from RGB values of pixels and plot theirs distribution for skin and nonskin examples

    # Y = .299*r + .587*g + .114*b # not needed
    # df['S'] = np.round(df.R + df.G + df.B).astype(int)
    #
    # print(0 in df.S)
    #
    # df['Cb'] = np.round(128 - .168736 * (255 * df.R / df.S) - .331364 * (255 * df.G / df.S) +
    #                     .5 * (255 * df.B / df.S)).astype(int)
    # df['Cr'] = np.round(128 + .5 * (255 * df.R / df.S) - .418688 * (255 * df.G / df.S) -
    #                     .081312 * (255 * df.B / df.S)).astype(int)

    df['Cb'] = np.round(128 - .168736 * df.R - .331364 * df.G +
                        .5 * df.B).astype(int)
    df['Cr'] = np.round(128 + .5 * df.R - .418688 * df.G -
                        .081312 * df.B).astype(int)
    # df['I'] = np.round(.596 * df.R - .275 * df.G -
    #                    .321 * df.B).astype(int)

    df.drop(['B', 'G', 'R'], axis=1, inplace=True)
    g = sns.catplot(data=pd.melt(df, id_vars='skin'),
                    x='variable', y='value', hue='variable',
                    col='skin', kind='box')
    # plt.show()
    # Separate the skin and nonskin training examples and fit two Gaussian
    # mixture models, one on the skin examples and another on the nonskin examples
    k = 4

    skin_data = df[df.skin == 1].drop(['skin'], axis=1).to_numpy()
    not_skin_data = df[df.skin == 2].drop(['skin'], axis=1).to_numpy()

    skin_gmm = GaussianMixture(n_components=k, covariance_type='full').fit(skin_data)
    not_skin_gmm = GaussianMixture(n_components=k, covariance_type='full').fit(not_skin_data)
    colors = ['navy', 'turquoise', 'darkorange', 'gold']

    # function to visualize the Gaussian mixture models fitted for skin and nonskin
    # fig = mpl.pyplot.figure()
    # axes = fig.add_axes([0, 0, 10, 10])
    # draw_ellipses(skin_gmm, axes)
    # plt.show()

    # Prediction

    # image_path = f"/home/{USERNAME}/Datasets/Site.png"
    # image_path = f"/home/{USERNAME}/Datasets/Larcc_dataset/astra/A/image0.png"
    # image_path = f"/home/{USERNAME}/Datasets/ASL/train/A/A2660.jpg"
    # image_path = f"/home/{USERNAME}/Datasets/Larcc_dataset/larcc_test_1/detected/A/image0.png"
    image_path = f"/home/{USERNAME}/Datasets/Larcc_dataset/larcc_test_1/A/image0.png"

    image = imread(image_path)[..., :3]
    cv2.imshow("Origial", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imshow("bilateralFilert", cv2.cvtColor(cv2.bilateralFilter(image, 15, 75, 75), cv2.COLOR_RGB2BGR))
    # cv2.imshow("Normalized", normalize_channel(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))

    image = cv2.bilateralFilter(image, 10, 20, 20)

    proc_image = np.reshape(rgb2ycbcr(image), (-1, 3))
    print(proc_image.shape)
    # proc_image1 = rgb2ycbcr(image)[:, :, 1:]
    # proc_image2 = rgb2yiq(image)[:, :, 1]
    # proc_image = np.reshape(cv2.merge([proc_image1, proc_image2]), (-1, 3))

    skin_score = skin_gmm.score_samples(proc_image[:, 1:])
    not_skin_score = not_skin_gmm.score_samples(proc_image[:, 1:])
    result = skin_score > not_skin_score

    result = result.reshape(image.shape[0], image.shape[1])
    result = np.bitwise_and(gray2rgb(255 * result.astype(np.uint8)), image)
    cv2.imshow("Result", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.waitKey()




