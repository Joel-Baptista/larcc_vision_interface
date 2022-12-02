import Augmentor
from vision_config.vision_definitions import ROOT_DIR
import os
import json

if __name__ == '__main__':
    aug_data = {}
    gestures = ["G1", "G2", "G5", "G6"]
    for subject in range(1, 6):

        aug_data[f"user{subject}"] = {}

        for gesture in gestures:

            if os.path.exists(ROOT_DIR + f"/Datasets/HANDS_dataset/Subject{subject}/hand_segmented/{gesture}/output"):
                os.rmdir(ROOT_DIR + f"/Datasets/HANDS_dataset/Subject{subject}/hand_segmented/{gesture}/output")

            res = os.listdir(ROOT_DIR + f"/Datasets/HANDS_dataset/Subject{subject}/hand_segmented/{gesture}")

            num_samples = int(2 * len(res))

            aug_data[f"user{subject}"][gesture] = num_samples

            p = Augmentor.Pipeline(source_directory=ROOT_DIR +
                f"/Datasets/HANDS_dataset/Subject{subject}/hand_segmented/{gesture}")

            p.rotate(probability=0.5,
                     max_left_rotation=25,
                     max_right_rotation=25)

            p.zoom(probability=0.25,
                   max_factor=1.3,
                   min_factor=1.1)

            p.sample(num_samples)

    print(aug_data)
    aug_data_json = json.dumps(aug_data, indent=4)

    with open(ROOT_DIR + "/Datasets/HANDS_dataset/aug_data.json", "w") as outfile:
        outfile.write(aug_data_json)
