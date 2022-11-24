import Augmentor
from vision_config.vision_definitions import ROOT_DIR


if __name__ == '__main__':

    for subject in range(1, 6):
        for gesture in range(1, 4):
            p = Augmentor.Pipeline(source_directory=ROOT_DIR +
                f"/Datasets/HANDS_dataset/Subject{subject}/hand_segmented/G{gesture}")

            p.rotate(probability=0.75,
                     max_left_rotation=25,
                     max_right_rotation=25)

            p.sample(1000)
