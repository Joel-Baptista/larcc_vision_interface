import os
from vision_config.vision_definitions import ROOT_DIR
import shutil

if __name__ == '__main__':
    if os.path.exists(ROOT_DIR + f"/Datasets/HANDS_dataset/training"):
        os.rmdir(ROOT_DIR + f"/Datasets/HANDS_dataset/training")

    if os.path.exists(ROOT_DIR + f"/Datasets/HANDS_dataset/validating"):
        os.rmdir(ROOT_DIR + f"/Datasets/HANDS_dataset/validating")

    sub_num = 5

    for subject in range(1, sub_num + 1):

        directories = os.listdir(ROOT_DIR + f"/Datasets/HANDS_dataset/Subject{subject}/hand_segmented")

        print(directories)

        for gesture in directories:
            pass