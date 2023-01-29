import os
import shutil
import random

if __name__=='__main__':
    
    test_split = 0.1
    val_plit = 0.2

    full_dataset_path = f"/home/{os.environ.get('USER')}/Datasets/Larcc_dataset/kinect_cropped"
    dataset_dir = f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect"
    labels = ["A", "F", "L", "Y"]
    directories = ["train", "val", "test"]

    for directory in directories:
        if os.path.exists(os.path.join(dataset_dir, f"{directory}")):
            shutil.rmtree(os.path.join(dataset_dir, f"{directory}"))
        
        os.makedirs(os.path.join(dataset_dir, f"{directory}"))

        for label in labels:
            os.makedirs(os.path.join(dataset_dir, f"{directory}", label))


    for label in labels:
        
        res = os.listdir(os.path.join(full_dataset_path, label))

        random.shuffle(res)

        test_index = int(len(res)*test_split)
        val_index = int(len(res)*val_plit + test_index)

        i = 0
        for file in res:

            if i <= test_index:
                shutil.copy2(os.path.join(full_dataset_path, label, file), os.path.join(dataset_dir, f"{directories[2]}", label, file))
            elif i <= val_index:
                shutil.copy2(os.path.join(full_dataset_path, label, file), os.path.join(dataset_dir, f"{directories[1]}", label, file))
            else:
                shutil.copy2(os.path.join(full_dataset_path, label, file), os.path.join(dataset_dir, f"{directories[0]}", label, file))

            i += 1