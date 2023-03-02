from networks import InceptionV3
import torch
import time
import copy
from torchvision import transforms
import os
import numpy as np
import csv
import argparse
from datasets import get_train_valid_loader
from torchsummary import summary
import json
# import matplotlib.pyplot as plt

# def imshow(inp):
#     """Imshow for Tensor."""
#     mean = np.array([0.5, 0.5, 0.5])
#     std = np.array([0.25, 0.25, 0.25])
#     inp = inp.numpy().transpose((1, 2, 0))
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     plt.show()


def main():

    parser = argparse.ArgumentParser(
                    prog = 'Pytorch Training',
                    description = 'It trains a Pytorch model')
    
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('-d', '--device', type=str, default="cuda:1", help='Decive used for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('-p', '--patience', type=int, default=None, help='Training patience')
    parser.add_argument('-a', '--augmentation', action='store_true', default=False, help='Augmentation')
    parser.add_argument('-v', '--version', type=str, default="", help='This string is added to the name of the model')
    parser.add_argument('-lt', '--load_train', action='store_true', default=False, help='Load train data')
    parser.add_argument('-td', '--train_dataset', type=str, default='train', help='Train dataset')
    parser.add_argument('-m', '--model', type=str, default="InceptionV3", help='Base model name')
    parser.add_argument('-luf', '--last_unfrozen', type=int, default=15, help='Last unfrozen layer')
    parser.add_argument('-cf', '--contrastive_features', type=int, default=16, help='Test dataset name')

    args = parser.parse_args()

    if args.patience is None:
        args.patience = args.epochs

    print("Script's arguments: ",args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Training device: ", device)

    paths = {"dataset": f'{os.getenv("HOME")}/Datasets/ASL/kinect/{args.train_dataset}',
        "test_dataset": f'{os.getenv("HOME")}/Datasets/ASL/kinect/multi_user',
        "results": f'{os.getenv("HOME")}/Datasets/ASL/kinect/results'}

    data_dir = f'{os.getenv("HOME")}/Datasets/ASL/kinect/'

    # unfreeze_layers = []
    unfreeze_layers = list(np.arange(args.last_unfrozen, 20))
    model = InceptionV3(4, args.learning_rate, unfreeze_layers= unfreeze_layers, device=device, class_features=2048,
                        con_features=args.contrastive_features, dropout=0.3, train_encoder=False)
    model.name = f"{args.model}"
    print("Model name: ", model.name)
    num_epochs = args.epochs
    if args.load_train:
        model.load_state_dict(torch.load(f"{data_dir}/results/{model.name}/{model.name}.pth"))
        print("Previous train loaded")

    model.name = f"{model.name}{args.version}"
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(299),
            # transforms.RandomResizedCrop(224, scale=(0.9, 1)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(299),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(299),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    if args.augmentation:
        data_transforms['train'] = transforms.Compose([
            transforms.Resize(299),
            # transforms.RandomResizedCrop(224, scale=(0.9, 1)),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomResizedCrop(299, scale=(0.7, 1.3))]), p=0.3),
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandAugment(magnitude=9)]), p=0.9),
            transforms.ToTensor(),
            # transforms.RandomErasing(p = 0.5, scale=(0.02, 0.1)),
            transforms.Normalize(mean, std)
        ])
    
    train_loader, val_loader, test_loader, dataset_sizes = get_train_valid_loader(
        os.path.join(data_dir, args.train_dataset), args.batch_size, data_transforms, None, test_path= None, shuffle=True, split=[0.2, 0.2])


    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    # Get a batch of training data
    # inputs, classes = next(iter(dataloaders['train']))

    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # imshow(out)

    # for child in model.children():
    #     for w in child.children():
    #         print(w)
    #         print("----------------")

    if not os.path.exists(os.path.join(data_dir, "results", f"{model.name}")):
        os.mkdir(os.path.join(data_dir, "results", f"{model.name}"))


    FILE = f"{model.name}.pth"
    history_collumns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    data_saving_path = os.path.join(data_dir, "results", f"{model.name}", "test_data")
    test_path = os.path.join(data_saving_path, f"test_results_{model.name}.csv")

    if not os.path.exists(data_saving_path):
        os.mkdir(data_saving_path)

    if len(unfreeze_layers) == 0:
        unfreeze_layers = [0]
    
    data = {"train_model": "train",
            "unfrozen_layers": float(unfreeze_layers[0]),
            "contrastive_features": float(model.con_features),
            "class_features": float(model.class_features),
            "dropout": float(model.drop_out),
            "learning_rate": float(model.learning_rate),
            "batch_size": float(args.batch_size),
            "epochs": float(args.epochs),
            "train_samples": dataset_sizes["train"],
            "val_samples": dataset_sizes["val"],
            "test_samples": dataset_sizes["test"],
            "augmentation": float(args.augmentation)
            }


    print(data)
    json_data = json.dumps(data)

    with open(data_saving_path + '/train_settings.json', 'w') as outfile:
        outfile.write(json_data)

    model.to(device)

    since = time.time()

    best_acc = 0.0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    early_stopping_counter = 0
    last_epoch_loss = np.inf
    early_stopping = False

    model.eval()

    inputs, labels = next(iter(dataloaders['train']))

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs, _ = model(inputs)
                    
    loss = model.loss(outputs, labels)

    print(f"Inital loss: {loss}")



    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            data_size = {"train": 0, "val": 0}

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs, _ = model(inputs)
                    
                    loss = model.loss(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        model.optimizer.zero_grad()
                        loss.backward()
                        model.optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                data_size[phase] += inputs.size(0)


            epoch_loss = running_loss / data_size[phase]
            history[f"{phase}_loss"].append(epoch_loss)

            epoch_acc = running_corrects.double() / data_size[phase]
            history[f"{phase}_acc"].append(epoch_acc.item())

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val' and epoch_loss > last_epoch_loss:
                early_stopping_counter += 1
                print("Counter: ", early_stopping_counter)
                if early_stopping_counter >= args.patience:
                    print('Early stopping!')
                    early_stopping = True
                    break

            elif phase == 'val' and epoch_loss <= last_epoch_loss:
                print("Reset counter")
                early_stopping_counter = 0

            if phase == 'val':
                last_epoch_loss = epoch_loss

        print()

        if (epoch + 1) % 10 == 0:
            print("Checkpoint - Saving training data")
            torch.save(best_model_wts, os.path.join(data_dir, "results", f"{model.name}", FILE)) 

            print("Data saved in : ", os.path.join(data_dir, "results", f"{model.name}"))
            with open(os.path.join(data_dir, "results", f"{model.name}", f"{model.name}_train.csv"), 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=history_collumns)
                writer.writeheader()
                for i in range(0, len(history["train_loss"])):

                    row = {"epoch": i+1, 
                        "train_loss": history[f"train_loss"][i], 
                        "train_acc": history[f"train_acc"][i],
                        "val_loss": history[f"val_loss"][i],
                        "val_acc": history[f"val_acc"][i]}
                    
                    writer.writerow(row)

        if early_stopping is True:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)


    torch.save(best_model_wts, os.path.join(data_dir, "results", f"{model.name}", FILE)) 

    with open(os.path.join(data_dir, "results", f"{model.name}", f"{model.name}_train.csv"), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=history_collumns)
        writer.writeheader()
        for i in range(0, len(history["train_loss"])):

            row = {"epoch": i+1, 
                   "train_loss": history[f"train_loss"][i], 
                   "train_acc": history[f"train_acc"][i],
                   "val_loss": history[f"val_loss"][i],
                   "val_acc": history[f"val_acc"][i]}
            
            writer.writerow(row)

    model.eval()
    test_labels = []
    test_preds = []
    running_corrects = 0
    count_test = 0
    for inputs, labels in dataloaders["test"]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

            count_test += inputs.size(0)
        
        for i in range(0, len(labels)):

            test_labels.append(labels[i].item())
            test_preds.append(preds[i].item())

    with open(test_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["labels", "predictions"])
        writer.writeheader()
        for i in range(0, len(test_preds)):

            row = {"labels": test_labels[i], 
                    "predictions": test_preds[i]}
            
            writer.writerow(row)


    test_acc = running_corrects.double() / count_test

    print('Test Accuracy of the model on the {:.0f} test images: {:.2f}'.format(count_test, 100 * test_acc))


if __name__ == '__main__':
    main()