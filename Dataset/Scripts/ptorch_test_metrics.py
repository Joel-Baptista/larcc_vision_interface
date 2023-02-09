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
    parser.add_argument('-m', '--model_name', type=str, default="InceptionV3_unfrozen", help='Model name')
    parser.add_argument('-t', '--test_dataset', type=str, default="kinect_test", help='Test dataset name')
    parser.add_argument('-c', '--contrastive', action="store_true", default=False, help="Train data is contrastive")


    args = parser.parse_args()
    
    model = args.model_name
    test  = args.test_dataset
    labels = ["A", "F", "L", "Y"]
    datasets = ["kinect_daniel", "kinect_manel", "kinect_lucas"]
    # datasets = []

    if len(datasets) == 0:
        datasets = [args.test_dataset]

    print("Testing: ", datasets)

    df_aux = pd.read_csv(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/{datasets[0]}/test_results_{model}.csv")

    df = pd.DataFrame(columns=df_aux.columns)

    for dataset in datasets:
        print(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/{dataset}/test_results_{model}.csv")
        df_aux = pd.read_csv(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/{dataset}/test_results_{model}.csv")    

        df = pd.concat([df, df_aux], axis=0, ignore_index=True)

    # df = pd.read_csv(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/{test}/test_results_{model}.csv")

    ground_truth = []
    predictions = []
    
    for i in range(0, len(df["labels"])):
        # print(df["predictions"][i])
        # print(i)
        ground_truth.append(labels[df["labels"][i]])
        predictions.append(labels[df["predictions"][i]])


    cm = confusion_matrix(ground_truth, predictions, labels=labels, normalize='true')
    cm = np.round(100 * cm, 2)
    blues = plt.cm.Blues
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=blues)


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


    if len(datasets) > 1:
        disp.figure_.savefig(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/{test}/cm_multi_user.png")
        with open(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/{test}/metrics_{model}_multi_user.json", 'w') as outfile:
            json.dump(dic_results, outfile)

    else:
        disp.figure_.savefig(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/{test}/cm.png")
        with open(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/{test}/metrics_{model}.json", 'w') as outfile:
            json.dump(dic_results, outfile)
    

    

    loss_lim = (0, 4)
    con_loss_lim = (8, 10)

    if args.plots:

        if args.contrastive:
            
            df = pd.read_csv(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}/train_results_{model}.csv")

            acc = df['train_acc']
            val_acc = df['val_acc']

            loss = df['train_loss']
            val_loss = df['val_loss']

            con_loss = df['train_con_loss']
            val_con_loss = df['val_con_loss']

            plt.figure(figsize=(8, 8))
            plt.subplot(3, 1, 1)
            plt.plot(acc, label='Training Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.ylabel('Accuracy')
            plt.ylim([min(plt.ylim()), 1])
            plt.title('Training and Validation Accuracy')

            plt.subplot(3, 1, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.ylabel('Cross Entropy')
            plt.ylim([loss_lim[0], loss_lim[1]])
            plt.title('Training and Validation Loss')
            plt.xlabel('epoch')

            plt.subplot(3, 1, 3)
            plt.plot(con_loss, label='Training Con Loss')
            plt.plot(val_con_loss, label='Validation Con Loss')
            plt.legend(loc='upper right')
            plt.ylabel('Constrative Cross Entropy')
            plt.ylim([con_loss_lim[0], con_loss_lim[1]])
            plt.title('Training and Validation Con Loss')
            plt.xlabel('epoch')


        else:
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
            plt.ylim([loss_lim[0], loss_lim[1]])
            plt.title('Training and Validation Loss')
            plt.xlabel('epoch')


        plt.savefig(f"/home/{os.environ.get('USER')}/Datasets/ASL/kinect/results/{model}//train_curves.png")
    
    plt.show()



