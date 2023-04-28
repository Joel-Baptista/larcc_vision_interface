import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, \
    precision_score, f1_score
import time

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def format_logits(logit):

    logit = logit.strip('][').split(' ')

    logit = [float(i) for i in logit if i != '']

    return logit

if __name__ == '__main__':


    labels = ["A", "F", "L", "Y", "NONE"]
    tresholds = [3, 3, 3, 3]
    soft_tresholds = [0.8, 0.8, 0.8, 0.8]

    # path_classes = f"/home/{os.environ.get('USER')}/results/InceptionV3/multi_user/test_results_InceptionV3.csv"
    # path_nones = f"/home/{os.environ.get('USER')}/results/InceptionV3/no_gesture/test_results_InceptionV3.csv"

    path_classes = "test_results1_InceptionV3.csv"
    path_nones = "test_results2_InceptionV3.csv"

    df1 = pd.read_csv(path_nones)
    df2 = pd.read_csv(path_classes)

    df1["labels"] = [4] * len(df1["labels"])

    df = pd.concat([df1, df2], axis=0, ignore_index=True)

    ground_truth = []
    predictions = []
    wrong_predictions = {}
    right_predictions = {}

    logits = {}
    confidences = {}

    tresholds =  np.linspace(0, 20, 50)
    t_best = [0] * 4
    f1_best = 0

    for i, t1 in enumerate(tresholds):
        for j, t2 in enumerate(tresholds):
            for k, t3 in enumerate(tresholds):
                for l, t4 in enumerate(tresholds):
                    
                    st = time.time()
                    tresh = [t1, t2, t3, t4]

                    ground_truth = []
                    predictions = []

                    for i in range(0, len(df["labels"])):

                        ground_truth.append(labels[df["labels"][i]])

                        p = df["predictions"][i]
                        l = format_logits(df["logits"][i])

                        if l[p] < tresh[p]:
                            p = 4

                        predictions.append(labels[p])


                    f1 = f1_score(ground_truth, predictions, average=None)
          
                    f1_mean = np.mean(f1)
                    
                    if f1_mean > f1_best:
                        t_best = tresh
                        f1_best = f1_mean
                        print(f"Found best case of f1 {f1_best} and treshold {t_best}") 
                    
                    print(time.time() - st)
                    