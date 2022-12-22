#!/usr/bin/env python3
from vision_config.vision_definitions import ROOT_DIR, USERNAME
import os
import json
import imgaug.augmenters as iaa
import imgaug as ia
import numpy as np
import cv2

if __name__ == '__main__':
    aug_data = {}
    dataset = "ASL"
    gestures = ["A", "F", "L", "Y"]
    # gestures = ["G1", "G2", "G5", "G6"]

    if not os.path.exists(f"/home/{USERNAME}/Datasets/{dataset}/augmented"):
        os.mkdir(f"/home/{USERNAME}/Datasets/{dataset}/augmented")

    for gesture in gestures:
        if not os.path.exists(f"/home/{USERNAME}/Datasets/{dataset}/augmented/{gesture}"):
            os.mkdir(f"/home/{USERNAME}/Datasets/{dataset}/augmented/{gesture}")

    print("Augmented Folder created")

    # ia.seed(1)

    seq = iaa.Sequential([
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.7))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.6, 1.4), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-10, 10),
            # shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order

    print("Augmentation sequence generated")

    for j in range(0, 2):
        for gesture in gestures:

            data_path = f"/home/{USERNAME}/Datasets/{dataset}/train/{gesture}"
            res = os.listdir(data_path)

            num_samples = int(2.5 * len(res))

            # Example batch of images.
            # The array has shape (32, 64, 64, 3) and dtype uint8.
            images = np.array(
                [cv2.imread(f"{data_path}/{file}") for file in res],
                dtype=np.uint8
            )

            # aug_data[f"user{subject}"][gesture] = num_samples
            images_aug = seq(images=images)

            for i, image in enumerate(images_aug):
                cv2.imwrite(f"/home/{USERNAME}/Datasets/{dataset}/augmented/{gesture}/image{j * len(res) + i}.png", image)

            print(f"Gesture {gesture} augmented")

    print(aug_data)
    aug_data_json = json.dumps(aug_data, indent=4)

    with open(f"/home/{USERNAME}/Datasets/{dataset}/aug_data.json", "w") as outfile:
        outfile.write(aug_data_json)
