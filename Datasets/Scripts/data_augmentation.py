import Augmentor
from vision_config.vision_definitions import ROOT_DIR


if __name__ == '__main__':

    for subject in range(1, 6):
        for gesture in range(1, 4):
            p = Augmentor.Pipeline(source_directory=ROOT_DIR +
                                    f"/Datasets/HANDS_dataset/Subject{subject}/Processed/G{gesture}")

            p.skew(probability=0.3)
            p.rotate_random_90(probability=0.3)

            p.sample(1000)
