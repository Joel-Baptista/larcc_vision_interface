import os
import cv2
import json
import torch
import numpy as np
from torchvision import transforms
import pandas as pd
from hand_classification.network.ptorch.networks import InceptionV3
from vision_config.vision_definitions import USERNAME, ROOT_DIR


if __name__ == '__main__':

    test_path = f"/home/{USERNAME}/Datasets/sliding_window"
    gestures = ["A", "F", "L", "Y", "None"]

    with open(f'{ROOT_DIR}/Dataset/configs/sliding_window.json') as f:
        config = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training device: ", device)

    model = InceptionV3(4, 0.0001, unfreeze_layers=list(np.arange(13, 20)), class_features=2048, device=device,
                con_features=16)
    model.name = "InceptionV3"

    trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{model.name}/{model.name}.pth', map_location=torch.device(device))
    model.load_state_dict(trained_weights)

    model.eval()

    model.to(device)

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])    

    # ground_truth = {}
    # logits = {}

    for hand in ["left", "right"]:
        res = os.listdir(f"{test_path}/{hand}")

        hand_classes = config[hand]
        # ground_truth[hand] = []
        # logits[hand] = []

        data = {"ground_truth": [],
                "logits": [],
                "image_name": []}

        num_list = []
        for file in res:

            num = int(''.join(filter(lambda i: i.isdigit(), file)))
            num_list.append(num)

        list1, list2 = zip(*sorted(zip(num_list, res)))

        hand_class = "None"

        for i, image_name in enumerate(list2):
            for key in hand_classes:
                if i in hand_classes[key]:
                    hand_class = key
            
            data["ground_truth"].append(hand_class)

            image = cv2.imread(test_path + "/" + hand + "/" + image_name)
            # print(test_path + "/" + image_name)

            cv2.putText(image, hand_class, ((0, 25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("hand", image)

            key = cv2.waitKey(1)

            if hand == "left":
                image = cv2.flip(image, 1)

            im_norm = data_transform(image).unsqueeze(0)
            im_norm = im_norm.to(device)

            with torch.no_grad():   
                outputs, _ = model(im_norm)
                _, preds = torch.max(outputs, 1)

            data["logits"].append(list(outputs.to("cpu").tolist())[0])
            data["image_name"].append(image_name)

            if key == ord('q'):
                break
        
        df = pd.DataFrame(data=data)
        df.to_csv(f"{hand}.csv")




