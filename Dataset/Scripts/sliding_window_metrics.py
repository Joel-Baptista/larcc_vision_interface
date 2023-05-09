import os
import cv2
import json
import torch
import numpy as np
from torchvision import transforms
import yaml
import pandas as pd
from hand_gesture_recognition.utils.hgr_utils import take_decision
from yaml.loader import SafeLoader
from vision_config.vision_definitions import USERNAME, ROOT_DIR

def format_logits(logit):

    logit = logit.strip('][').split(',')

    logit = [float(i) for i in logit if i != '']

    return logit

if __name__ == '__main__':
    model_name = "InceptionV3"
    n_frames = 10
    min_coef = 0.5

    with open(f'{ROOT_DIR}/hand_gesture_recognition/config/model/{model_name}.yaml') as f:
        data = yaml.load(f, Loader=SafeLoader)
        print(data)

    with open(f'{ROOT_DIR}/hand_gesture_recognition/config/model/thresholds.yaml') as f:
        t = yaml.load(f, Loader=SafeLoader)
        print(t)

    thresholds = t["thresholds"][data["threshold_choice"]]
    cm = t["confusion_matrix"][data["threshold_choice"]]
    print(thresholds)
    print(cm)

    test_path = f"/home/{USERNAME}/Datasets/sliding_window"
    gestures = ["A", "F", "L", "Y", "None"]

    with open(f'{ROOT_DIR}/Dataset/configs/sliding_window.json') as f:
        config = json.load(f)

    for hand in ["left", "right"]:
        
        buffer = [4] * n_frames # Initializes the buffer with 5 "NONE" gestures
        df = pd.read_csv(f"{hand}.csv")
        print(df)

        for i in range(0, len(df["image_name"])):
            logits = format_logits(df["logits"][i])
            preds = logits.index(max(logits))
            
            pred, confid, buffer = take_decision([logits], preds, thresholds, buffer, cm, min_coef=min_coef)

            #TODO save grounth thruth and make confusion matrix
