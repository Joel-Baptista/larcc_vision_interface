from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, \
    precision_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                prog = 'Pytorch Training',
                description = 'It trains a Pytorch model')

    parser.add_argument('-p', '--plots', action='store_true', default=False, help='Plot train curves')
    args = parser.parse_args()
    
    model = "InceptionV3_unfrozen_aug"
    test  = "test"
    labels = ["A", "F", "L", "Y"]

    df = pd.read_csv(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/{test}/test_results_{model}.csv")
    
    ground_truth = []
    predictions = []
    for i in range(0, len(df["labels"])):

        ground_truth.append(labels[df["labels"][i]])
        predictions.append(labels[df["predictions"][i]])


    cm = confusion_matrix(ground_truth, predictions, labels=labels, normalize='true')
    cm = np.round(100 * cm, 2)
    blues = plt.cm.Blues
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=blues)
    disp.figure_.savefig(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/{test}/cm.png")
    


    recall = recall_score(ground_truth, predictions, average=None)
    precision = precision_score(ground_truth, predictions, average=None)
    f1 = f1_score(ground_truth, predictions, average=None)

    dic_results = {"accuracy": accuracy_score(ground_truth, predictions), "precision": {},
                   "recall": {}, "f1": {}}

    for i, label in enumerate(labels):
        dic_results["recall"][label] = recall[i]
        dic_results["precision"][label] = precision[i]
        dic_results["f1"][label] = f1[i]

    dic_results["recall"]["average"] = np.mean(recall)
    dic_results["precision"]["average"] = np.mean(precision)
    dic_results["f1"]["average"] = np.mean(f1)

    print(dic_results)

    with open(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/{test}/metrics_{model}.json", 'w') as outfile:
        json.dump(dic_results, outfile)

    if args.plots:
        df = pd.read_csv(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/train_results_{model}.csv")

        acc = df['train_acc']
        val_acc = df['val_acc']

        loss = df['train_loss']
        val_loss = df['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0, 1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')


        plt.savefig(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/{test}/train_curves.png")
    
    plt.show()



