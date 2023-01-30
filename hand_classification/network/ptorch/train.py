from networks import InceptioV3_frozen, InceptioV3_unfrozen
import torch
import time
import copy
import torchvision
from torchvision import datasets, transforms
import os
import numpy as np
import csv
import argparse


def main():

    parser = argparse.ArgumentParser(
                    prog = 'Pytorch Training',
                    description = 'It trains a Pytorch model')
    
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('-d', '--device', type=str, default="cuda:1", help='Decive used for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('-p', '--patience', type=int, default=None, help='Training patience')
    

    args = parser.parse_args()

    if args.patience is None:
        args.patience = args.epochs

    print("Script's arguments: ",args)
    data_dir = f'{os.getenv("HOME")}/Datasets/ASL/kinect'

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Training device: ", device)

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

    data_dir = f'{os.getenv("HOME")}/Datasets/ASL/kinect'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                shuffle=True, num_workers=2, prefetch_factor=1)
                        for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    print("Training classes: ",class_names)

    model = InceptioV3_unfrozen(4, args.learning_rate)
    num_epochs = args.epochs

    # for child in model.children():
    #     print(child)
    #     print("----------------")
    if not os.path.exists(os.path.join(data_dir, "results", f"{model.name}")):
        os.mkdir(os.path.join(data_dir, "results", f"{model.name}"))


    FILE = f"{model.name}.pth"
    history_collumns = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
    history_path = os.path.join(data_dir, "results", f"{model.name}",f"train_results_{model.name}.csv")
    test_path = os.path.join(data_dir, "results", f"{model.name}", f"test_results_{model.name}.csv")

    model.to(device)

    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    early_stopping_counter = 0
    last_epoch_loss = np.inf
    early_stopping = False

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

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # print(outputs)
                    # print(labels)
                    loss = model.loss(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        model.optimizer.zero_grad()
                        loss.backward()
                        model.optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
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

        if epoch % 10 == 0:
            print("Checkpoint - Saving training data")
            torch.save(best_model_wts, os.path.join(data_dir, "results", f"{model.name}", FILE)) 

            with open(history_path, 'w') as csvfile:
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

    with open(history_path, 'w') as csvfile:
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
    for inputs, labels in dataloaders["test"]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)
        
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


    test_acc = running_corrects.double() / dataset_sizes["test"]

    print('Test Accuracy of the model on the {:.0f} test images: {:.2f}'.format(dataset_sizes["test"], 100 * test_acc))


if __name__ == '__main__':
    main()