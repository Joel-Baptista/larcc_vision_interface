import Augmentor
from vision_config.vision_definitions import ROOT_DIR, USERNAME
import os
import json

if __name__ == '__main__':
    aug_data = {}
    dataset = "ASL"
    gestures = ["A", "F", "L", "Y"]
    # gestures = ["G1", "G2", "G5", "G6"]

    for gesture in gestures:
        if os.path.exists(f"/home/{USERNAME}/Datasets/{dataset}/augmented/{gesture}"):
            os.rmdir(f"/home/{USERNAME}/Datasets/{dataset}/augmented/{gesture}")
            os.mkdir(f"/home/{USERNAME}/Datasets/{dataset}/augmented/{gesture}")

    # for subject in range(1, 5):

        # aug_data[f"user{subject}"] = {}

    for gesture in gestures:

        res = os.listdir(f"/home/{USERNAME}/Datasets/{dataset}/train/{gesture}")

        num_samples = int(2.5 * len(res))

        # aug_data[f"user{subject}"][gesture] = num_samples

        p = Augmentor.Pipeline(source_directory=f"/home/{USERNAME}/Datasets/{dataset}/train/{gesture}",
                               output_directory=f"/home/{USERNAME}/Datasets/{dataset}/augmented/{gesture}")

        p.rotate(probability=0.75,
                 max_left_rotation=15,
                 max_right_rotation=15)

        # p.zoom(probability=0.25,
        #        min_factor=0.9,
        #        max_factor=1.1)

        # p.zoom_random(probability=0.25,
        #               percentage_area=0.8,
        #               randomise_percentage_area=False)

        p.random_brightness(probability=0.25,
                            min_factor=0.9,
                            max_factor=1.1)

        p.random_color(probability=0.25,
                       min_factor=0.2,
                       max_factor=1.0)

        p.random_contrast(probability=0.25,
                          min_factor=0.2,
                          max_factor=1.0)

        p.random_erasing(probability=0.25,
                         rectangle_area=0.4)

        p.sample(num_samples)

    print(aug_data)
    aug_data_json = json.dumps(aug_data, indent=4)

    with open(f"/home/{USERNAME}/Datasets/{dataset}/aug_data.json", "w") as outfile:
        outfile.write(aug_data_json)
