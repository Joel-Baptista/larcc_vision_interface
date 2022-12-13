import Augmentor
from vision_config.vision_definitions import ROOT_DIR
import os
import json

if __name__ == '__main__':
    aug_data = {}
    dataset = "ASL"
    gestures = ["A", "F", "L", "Y"]
    # gestures = ["G1", "G2", "G5", "G6"]

    for gesture in gestures:

        if os.path.exists(ROOT_DIR + f"/Datasets/{dataset}/augmented/{gesture}"):
            os.rmdir(ROOT_DIR + f"/Datasets/{dataset}/augmented/{gesture}")
            os.mkdir(ROOT_DIR + f"/Datasets/{dataset}/augmented/{gesture}")

    # for subject in range(1, 5):

        # aug_data[f"user{subject}"] = {}

    for gesture in gestures:

        res = os.listdir(ROOT_DIR + f"/Datasets/{dataset}/train/{gesture}")

        num_samples = int(2 * len(res))

        # aug_data[f"user{subject}"][gesture] = num_samples

        p = Augmentor.Pipeline(source_directory=ROOT_DIR + f"/Datasets/{dataset}/train/{gesture}",
                               output_directory=ROOT_DIR + f"/Datasets/{dataset}/augmented/{gesture}")

        p.rotate(probability=0.25,
                 max_left_rotation=25,
                 max_right_rotation=25)

        p.zoom(probability=0.25,
               min_factor=0.8,
               max_factor=1.2)

        p.zoom_random(probability=0.25,
                      percentage_area=0.7,
                      randomise_percentage_area=False)

        # p.random_brightness(probability=0.25,
        #                     min_factor=0.9,
        #                     max_factor=1.1)
        #
        # p.random_color(probability=0.25,
        #                min_factor=0.9,
        #                max_factor=1.1)

        p.sample(num_samples)

    print(aug_data)
    aug_data_json = json.dumps(aug_data, indent=4)

    with open(ROOT_DIR + f"/Datasets/{dataset}/aug_data.json", "w") as outfile:
        outfile.write(aug_data_json)
