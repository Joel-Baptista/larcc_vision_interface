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
import random

def format_logits(logit):

    logit = logit.strip('][').split(',')

    logit = [float(i) for i in logit if i != '']

    return logit

def buffer_analyses(buffer, cm, min_coef):
    probability = []
    confidance = []
    coeficients = np.linspace(min_coef, 1.0, num=len(buffer))
    avg_coeficients = sum(coeficients) / len(coeficients)

    for i in range(0, 5):

        prob = 0

        for j, prediction in enumerate(buffer):

            prob = prob + (cm[i][prediction] * coeficients[j]) / (100 * len(buffer)) # Weighted Average of probabilities

        probability.append(prob)
        confidance.append(prob / (cm[i][i] * avg_coeficients) * 100)

    pred = probability.index(max(probability))
    confid = confidance[pred]


    return pred, round(confid, 4), buffer

def main():
    model_name = "InceptionV3"
    is_sliding_ground_truth = False

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

    threshold_choice = ["t_p", "t_f1"]
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
    fps_test = 5

    jump = fps_test / fps_aquisition
    analyze_count=0

    results = {"score": [],
               "n_frames": [],
               "min_coef": []}

    for n_f in n_frames:
        for m_c in min_coef:
            ground_truths = []
            predictions = []
            count += 1
            print(f"Iteration {count} out of {iterations}")

            if n_f == 1:
                m_c = 1

            for hand in ["left", "right"]:
                
                buffer = [4] * n_f
                ground_truths_buffer = [4] * n_f
                df = pd.read_csv(f"{hand}.csv")

                for i in range(0, len(df["image_name"])):

                    r = random.random()
                    
                    if r > jump:
                        continue
                
                    ground_truths_buffer.pop(0)
                    ground_truths_buffer.append(gestures.index(df["ground_truth"][i]))

                    analyze_count +=1
                    logits = format_logits(df["logits"][i])
                    preds = logits.index(max(logits))
                    
                    pred, confid, buffer = take_decision([logits], preds, thresholds, buffer, cm, min_coef=m_c)


                    if is_sliding_ground_truth:
                        ground_truth, _, ground_truths_buffer = buffer_analyses(ground_truths_buffer, cm, min_coef=m_c)
                    else:

                        ground_truth = gestures.index(df["ground_truth"][i])
                        
                    ground_truths.append(gestures[ground_truth])

                    predictions.append(gestures[pred])

            f1 = precision_score(ground_truths, predictions, average=None)

            f1_mean = np.mean(f1)

            results["score"].append(f1_mean)
            results["min_coef"].append(m_c)
            results["n_frames"].append(n_f)
            
            if f1_mean > f1_best:
                comb = (n_f, m_c)
                f1_best = f1_mean
                gt_best = ground_truths
                p_best = predictions
                print(f"Found best case of f1 {f1_best} and treshold {comb}") 

    df = pd.DataFrame(data=results)

    print(df)
    df = df.sort_values(by=['score'], ascending=False)
    print(df)
    if is_sliding_ground_truth: 
        df.to_csv(f"{ROOT_DIR}/Dataset/results/{fps_test}_slide_{threshold_choice[data['threshold_choice']]}.csv")
    else:
        df.to_csv(f"{ROOT_DIR}/Dataset/results/{fps_test}_no_slide_{threshold_choice[data['threshold_choice']]}.csv")

    print(f"Best Score is {f1_best} with n_frames {comb[0]} and min_coef {comb[1]}")
    print(analyze_count)
    cm_pred = confusion_matrix(gt_best, p_best, labels=gestures, normalize='pred')
    cm_pred = np.round(100 * cm_pred, 2)
    blues = plt.cm.Blues
    disp_pred = ConfusionMatrixDisplay(confusion_matrix=cm_pred, display_labels=gestures)
    disp_pred.plot(cmap=blues, values_format='')
    plt.title(f'{thresholds} - Prediction Normalized')

    cm_true = confusion_matrix(gt_best, p_best, labels=gestures, normalize='true')
    cm_true = np.round(100 * cm_true, 2)
    blues = plt.cm.Blues
    disp_true = ConfusionMatrixDisplay(confusion_matrix=cm_true, display_labels=gestures)
    disp_true.plot(cmap=blues, values_format='')
    plt.title(f'{thresholds} - Ground Truth Normalized')    

    plt.show()



if __name__ == '__main__':
    main()