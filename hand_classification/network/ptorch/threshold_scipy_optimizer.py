import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, \
    precision_score, f1_score
import time
from scipy.optimize import minimize 
import json

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def format_logits(logit):

    logit = logit.strip('][').split(' ')

    logit = [float(i) for i in logit if i != '']

    return logit

def calculate_f1_score(tresh, df, labels):

    ground_truth = []
    predictions = []
    print(tresh)
    for i in range(0, len(df["labels"])):

        ground_truth.append(labels[df["labels"][i]])

        p = df["predictions"][i]
        l = format_logits(df["logits"][i])

        if l[p] < tresh[p]:
            p = 4

        predictions.append(labels[p])


    f1 = f1_score(ground_truth, predictions, average=None)

    return 1 - np.mean(f1)
    


if __name__ == '__main__':


    labels = ["A", "F", "L", "Y", "NONE"]
    tresholds = [3, 3, 3, 3]
    soft_tresholds = [0.8, 0.8, 0.8, 0.8]

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
    methods  = ["Nelder-Mead", "Powell", "COBYLA"]

    for method in methods:
        for i in range(0, 10):
            x0 = 15 * np.random.rand(4)
            print(x0)

            res = minimize(calculate_f1_score, x0= x0, args=(df, labels), method=method)
            print(dict(res))

            result = {"x":list(res["x"]), "y":res["fun"]}

            with open(f'./results/{method}_{i}.json', 'w') as fp:
                json.dump(result, fp)

    

