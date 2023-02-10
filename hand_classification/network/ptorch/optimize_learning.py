import torch
import os
import numpy as np
import csv
import argparse
import torch.nn as nn
from datasets import get_train_valid_loader
from losses import SupConLoss, SimpleConLoss
from funcs import set_transforms, test, train
from networks import InceptionV3
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                prog = 'Pytorch Training',
                description = 'It trains a Pytorch model')
    
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('-d', '--device', type=str, default="cuda:1", help='Decive used for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('-p', '--patience', type=int, default=None, help='Training patience')
    parser.add_argument('-a', '--augmentation', action='store_true', default=False, help='Augmentation')
    parser.add_argument('-v', '--version', type=str, default="", help='This string is added to the name of the model')
    parser.add_argument('-lt', '--load_train', action='store_true', default=False, help='Load train data')
    parser.add_argument('-td', '--train_dataset', type=str, default='train', help='Train dataset')
    parser.add_argument('-t', '--temperature', type=float, default=0.07, help='Temperature for contrastive loss')
    parser.add_argument('-ver', '--verbose', type=str, default=0, help='Select verbose mode')

    args = parser.parse_args()

    if args.patience is None:
        args.patience = args.epochs

    # last_unfrozen = [13, 15, 17]
    last_unfrozen = [15]
    # learning_rate = [0.001, 0.0001, 0.00001]
    learning_rate = [0.0001]
    # batch_size = [64, 128, 200]
    batch_size = [200]
    scale = [0.3, 0.5, 0.8]
    prob_scale = [0.3, 0.5, 0.8]
    # con_features = [16, 32, 64]
    con_features = [32]
    drop_out = [0.0, 0.3, 0.5]

    print("Script's arguments: ",args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Training device: ", device)

    paths = {"dataset": f'{os.getenv("HOME")}/Datasets/ASL/kinect/{args.train_dataset}',
             "test_dataset": f'{os.getenv("HOME")}/Datasets/ASL/kinect/multi_user',
             "results": f'{os.getenv("HOME")}/Datasets/ASL/kinect/results'}

    class_loss = nn.CrossEntropyLoss()

    optimized_params = {"1st": {"f1": 0}, "2nd": {}, "3rd": {}}
    field_names = ["batch_size", "scale", "prob_scale", "unfrozen", "learning_rate", "drop_out", "con_features", "f1", "acc"]

    for bs in batch_size:
        for sc in scale:
            for psc in prob_scale:

                data_transforms = set_transforms(args, s=sc, p=psc)

                train_loader, val_loader, test_loader, dataset_sizes = get_train_valid_loader(
                    paths["dataset"], bs, data_transforms, None, test_path= paths["test_dataset"], shuffle=True, split=[0.2, 0.3])
                
                dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader, "dataset_sizes": dataset_sizes}

                for last in last_unfrozen:
                    for lr in learning_rate:
                        for drop in drop_out:
                            for cf in con_features:

                                params = {"batch_size": bs,
                                          "scale": sc,
                                          "prob_scale": psc,
                                          "unfrozen": last,
                                          "learning_rate": lr,
                                          "drop_out": drop,
                                          "con_features": cf,
                                          "f1": 0,
                                          "acc": 0}

                                print("Testing with params: ", params)

                                model = InceptionV3(4, lr, device=device, unfreeze_layers= list(np.arange(last, 18)), dropout=drop, con_features=cf)
                                model.name = f"{model.name}_aux"
                                model.to(device)

                                best_ws = train(model, dataloaders, paths, args, device)

                                model.load_state_dict(best_ws)

                                f1, acc = test(model, dataloaders, paths, device)

                                print('Test Accuracy is {:.2f} and {:.2f}'.format(acc.item(), f1))

                                if f1 > optimized_params["1st"]["f1"]:
                                    print("Found new best combination!")
                                    optimized_params["3rd"] = optimized_params["2nd"]
                                    optimized_params["2nd"] = optimized_params["1st"]
                                    optimized_params["1st"] = params

                                    with open(os.path.join(paths["results"], f'{model.name}', "optimized.csv"), 'w') as csvfile:
                                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                                        writer.writeheader()
                                        for key in optimized_params.keys():

                                            row = optimized_params[key]
                                            
                                            writer.writerow(row)
                                elif f1 > optimized_params["2nd"]["f1"]:
                                    print("Found new second best combination!")
                                    optimized_params["3rd"] = optimized_params["2nd"]
                                    optimized_params["2nd"] = params

                                    with open(os.path.join(paths["results"], f'{model.name}', "optimized.csv"), 'w') as csvfile:
                                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                                        writer.writeheader()
                                        for key in optimized_params.keys():

                                            row = optimized_params[key]
                                            
                                            writer.writerow(row)

                                elif f1 > optimized_params["3rd"]["f1"]:
                                    print("Found new second best combination!")
                                    optimized_params["3rd"] = params

                                    with open(os.path.join(paths["results"], f'{model.name}', "optimized.csv"), 'w') as csvfile:
                                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                                        writer.writeheader()
                                        for key in optimized_params.keys():

                                            row = optimized_params[key]
                                            
                                            writer.writerow(row)
