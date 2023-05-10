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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, \
    precision_score, f1_score
import matplotlib.pyplot as plt

def format_logits(logit):

    logit = logit.strip('][').split(',')

    logit = [float(i) for i in logit if i != '']

    return logit

if __name__ == '__main__':
    model_name = "InceptionV3"
    n_frames = [x for x in range(1, 11)]
    print(n_frames)
    min_coef = np.linspace(0, 1, 11)
    iterations = len(n_frames) * len(min_coef)

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

    f1_best = 0
    comb = (0, 0)
    count = 0
    fps_aquisition = 30
    fps_test = 10

    jump = int(fps_aquisition / fps_test)
    frame_count = 0
    analyze_count=0

    for n_f in n_frames:
        for m_c in min_coef:
            ground_truths = []
            predictions = []
            count += 1
            print(f"Iteration {count} out of {iterations}")

            for hand in ["left", "right"]:
                
                buffer = [4] * n_f
                df = pd.read_csv(f"{hand}.csv")

                for i in range(0, len(df["image_name"])):
                    
                    frame_count += 1
                    if frame_count < jump:

                        continue
                    else:
                        frame_count = 0

                    analyze_count +=1
                    logits = format_logits(df["logits"][i])
                    preds = logits.index(max(logits))
                    
                    pred, confid, buffer = take_decision([logits], preds, thresholds, buffer, cm, min_coef=m_c)

                    ground_truths.append(df["ground_truth"][i])
                    predictions.append(gestures[pred])

            f1 = precision_score(ground_truths, predictions, average=None)

            f1_mean = np.mean(f1)
            
            if f1_mean > f1_best:
                comb = (n_f, m_c)
                f1_best = f1_mean
                gt_best = ground_truths
                p_best = predictions
                print(f"Found best case of f1 {f1_best} and treshold {comb}") 

    print(f"Best Score is {f1_best} with n_frames {comb[0]} and min_coef {comb[1]}")
    print(analyze_count)
    cm_pred = confusion_matrix(ground_truths, predictions, labels=gestures, normalize='pred')
    cm_pred = np.round(100 * cm_pred, 2)
    blues = plt.cm.Blues
    disp_pred = ConfusionMatrixDisplay(confusion_matrix=cm_pred, display_labels=gestures)
    disp_pred.plot(cmap=blues, values_format='')
    plt.title(f'{thresholds} - Prediction Normalized')

    cm_true = confusion_matrix(ground_truths, predictions, labels=gestures, normalize='true')
    cm_true = np.round(100 * cm_true, 2)
    blues = plt.cm.Blues
    disp_true = ConfusionMatrixDisplay(confusion_matrix=cm_true, display_labels=gestures)
    disp_true.plot(cmap=blues, values_format='')
    plt.title(f'{thresholds} - Ground Truth Normalized')    

    plt.show()

