import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, \
    precision_score, f1_score

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
    # tresholds = [0.41, 3.76, 5.81, 6.34] # F1 Optimized
    tresholds = [3.11, 8.04, 8.15, 8.25] # Precision Optimized

    soft_tresholds = [-1] * 4

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

    for label in labels:
        print(label)
        confidences[label] = []
        logits[label] = []
        wrong_predictions[label] = []
        right_predictions[label] = []

    for i in range(0, len(df["labels"])):

        ground_truth.append(labels[df["labels"][i]])

        p = df["predictions"][i]
        l = format_logits(df["logits"][i])
        c = softmax(l) 

        if l[p] < tresholds[p]:
            p = 4
        elif c[p] < soft_tresholds[p]:
            p = 4
         
        logits[labels[df["labels"][i]]].append(l)

        confidences[labels[df["labels"][i]]].append(c)

        predictions.append(labels[p])


    n_bins = 40

    # fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

    fig, axs = plt.subplots(nrows=2, ncols=3)
    colors = ['red', 'tan', 'lime', "blue"]

    # print(np.array(logits[label[0]]).T.tolist())

    for i, ax_col in enumerate(axs):
        for j, ax_lin in enumerate(ax_col):
            print(ax_lin)

            idx = j + 3 * i

            if idx >= 5:
                break

            ax_lin.hist(np.array(logits[labels[idx]]).T.tolist(), n_bins, density=True, histtype='bar', stacked=False)
            print(idx)
            print(labels[idx])
            ax_lin.set_title(labels[idx])
            ax_lin.legend(labels)

            if idx < 4:
                ax_lin.axvline(tresholds[idx], color='y', linewidth= 3)
        
    fig.tight_layout()
    # plt.savefig(f"/home/{os.environ.get('USER')}/results/InceptionV3/{test}/logits_hist.png")
    # plt.savefig(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/{test}/logits_hist.png")

    fig1, axs1 = plt.subplots(nrows=2, ncols=3)

    for i, ax_col in enumerate(axs1):
        for j, ax_lin in enumerate(ax_col):

            idx = j + 3 * i

            if idx >= 5:
                break

            ax_lin.hist(np.array(confidences[labels[idx]]).T.tolist(), n_bins, density=True, histtype='bar', stacked=False)
            ax_lin.set_title(labels[idx])
            ax_lin.legend(labels)

            if idx < 4:
                ax_lin.axvline(soft_tresholds[idx], color='y', linewidth= 2)
        
    fig1.tight_layout()

    cm_pred = confusion_matrix(ground_truth, predictions, labels=labels, normalize='pred')
    cm_pred = np.round(100 * cm_pred, 2)
    blues = plt.cm.Blues
    disp_pred = ConfusionMatrixDisplay(confusion_matrix=cm_pred, display_labels=labels)
    disp_pred.plot(cmap=blues, values_format='')
    plt.title(f'{tresholds} - Prediction Normalized')

    cm_true = confusion_matrix(ground_truth, predictions, labels=labels, normalize='true')
    cm_true = np.round(100 * cm_true, 2)
    blues = plt.cm.Blues
    disp_true = ConfusionMatrixDisplay(confusion_matrix=cm_true, display_labels=labels)
    disp_true.plot(cmap=blues, values_format='')
    plt.title(f'{tresholds} - Ground Truth Normalized')    

    plt.show()