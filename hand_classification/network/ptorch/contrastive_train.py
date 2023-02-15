from networks import InceptioV3_frozen, InceptioV3_unfrozen, InceptionV3
import torch
import time
import copy
import torchvision
from torchvision import datasets, transforms
import os
import numpy as np
import csv
import argparse
from funcs import test
import torch.nn as nn
from datasets import get_train_valid_loader
from losses import SupConLoss, SimpleConLoss
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

def update_contrastive(self, raw_state_batch, raw_state_batch_pos):
    
    z_a = self.encode_image(raw_state_batch)
    z_pos = self.encode_image(raw_state_batch_pos, target=True)
    
    logits = self.curl.compute_logits(z_a=z_a, z_pos=z_pos)
    labels = torch.arange(logits.shape[0]).long().to(self.device)

    loss = self.curl.loss(logits, labels)
    
    self.image_encoder.optimizer.zero_grad(set_to_none=True)
    self.curl.optimizer.zero_grad(set_to_none=True)
    
    loss.backward()
    
    print(loss.item())
    
    self.image_encoder.optimizer.step()
    self.curl.optimizer.step()


def main():

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

    args = parser.parse_args()

    paths = {"dataset": f'{os.getenv("HOME")}/Datasets/ASL/kinect/{args.train_dataset}',
            "test_dataset": f'{os.getenv("HOME")}/Datasets/ASL/kinect/multi_user',
            "results": f'{os.getenv("HOME")}/Datasets/ASL/kinect/results'}

    if args.patience is None:
        args.patience = args.epochs

    print("Script's arguments: ",args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Training device: ", device)

    data_dir = f'{os.getenv("HOME")}/Datasets/ASL/kinect/'

    model = InceptionV3(4, args.learning_rate, unfreeze_layers= list(np.arange(13, 19)), device=device, con_features=16, dropout=0.3)
    model.name = f"{model.name}{args.version}"
    num_epochs = args.epochs

    if args.load_train:
        model.load_state_dict(torch.load(f"{data_dir}/results/{model.name}/{model.name}.pth"))
        
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
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomResizedCrop(299, scale=(0.7, 1.3)), transforms.RandAugment(magnitude=18)]), p=0.7),
            transforms.ToTensor(),
            transforms.RandomErasing(p = 0.5, scale=(0.02, 0.1)),
            transforms.Normalize(mean, std)
        ])
        # data_transforms['train'] = transforms.Compose([
        #     transforms.Resize(299),
        #     # transforms.RandomResizedCrop(224, scale=(0.9, 1)),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.RandomApply(
        #         torch.nn.ModuleList([
        #             transforms.RandomResizedCrop(299, scale=(0.7, 1.3)),
        #             transforms.RandomRotation()
        #             ],), p=0.5),
        #     transforms.RandAugment(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        # ])
    
    train_loader, val_loader, test_loader, dataset_sizes = get_train_valid_loader(
        os.path.join(data_dir, args.train_dataset), args.batch_size, data_transforms, None, test_path= paths["test_dataset"], shuffle=True, split=[0.2, 0.2])


    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader, "dataset_sizes": dataset_sizes}
    # # Get a batch of training data
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
    history_collumns = ["epoch", "train_loss", "val_loss", 'train_con_loss', 'val_con_loss', 'train_acc', 'val_acc']
    data_saving_path = os.path.join(data_dir, "results", f"{model.name}", "test_data")

    if not os.path.exists(data_saving_path):
        os.mkdir(data_saving_path)

    model.to(device)

    since = time.time()

    best_loss = 0.0
    
    history = {'train_loss': [], 'val_loss': [], 'train_con_loss': [], 'val_con_loss': [], 'train_acc': [], 'val_acc': []}
    early_stopping_counter = 0
    last_epoch_loss = np.inf
    early_stopping = False

    # model.eval()

    # inputs, labels = next(iter(dataloaders['train']))

    # inputs = inputs.to(device)
    # labels = labels.to(device)

    # outputs, hooks = model(inputs)
                    
    # con_loss = model.loss(hooks[output_layer].unsqueeze(2), labels)
    # loss_final = class_loss(outputs, labels) 

    # print(f"Inital contrastive loss: {con_loss}")
    # print(f"Inital classification loss: {loss_final}")

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
            running_con_loss = 0.0
            running_corrects = 0
            
            # dataloaders[phase].shuffle()
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs, contrastive_features = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss_con = model.con_loss(contrastive_features.unsqueeze(2), labels) # SupConLoss
                    loss= model.loss(outputs, labels) 

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        model.optimizer.zero_grad()

                        loss_con.backward(retain_graph=True)
    
                        loss.backward()
                        
                        model.optimizer.step()


                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_con_loss += loss_con.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_con_loss = running_con_loss / dataset_sizes[phase]
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_con_loss"].append(epoch_con_loss)

            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            history[f"{phase}_acc"].append(epoch_acc.item())


            print('{} Loss: {:.4f} - Con Loss: {:.4f} - Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_con_loss, epoch_acc))

            # deep copy the model

            if phase == 'train' and epoch_loss < best_loss:
                best_loss = epoch_loss

            if phase == 'val' and epoch_loss > last_epoch_loss:
                early_stopping_counter += 1
                print("Counter: ", early_stopping_counter)
                if early_stopping_counter >= args.patience:
                    print('Early stopping!')
                    early_stopping = True
                    break

            elif phase == 'val' and epoch_loss <= last_epoch_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
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
                        "val_loss": history[f"val_loss"][i],
                        "train_con_loss": history[f"train_con_loss"][i],
                        "val_con_loss": history[f"val_con_loss"][i],
                        "train_acc": history[f"train_acc"][i],
                        "val_acc": history[f"val_acc"][i]}
                    
                    writer.writerow(row)

        if early_stopping is True:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)


    torch.save(best_model_wts, os.path.join(data_dir, "results", f"{model.name}", FILE)) 

    with open(os.path.join(data_dir, "results", f"{model.name}", f"{model.name}_train.csv"), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=history_collumns)
        writer.writeheader()
        for i in range(0, len(history["train_loss"])):

            row = {"epoch": i+1, 
                "train_loss": history[f"train_loss"][i], 
                "val_loss": history[f"val_loss"][i],
                "train_con_loss": history[f"train_con_loss"][i],
                "val_con_loss": history[f"val_con_loss"][i],
                "train_acc": history[f"train_acc"][i],
                "val_acc": history[f"val_acc"][i]}
            
            writer.writerow(row)

    f1, acc = test(model, dataloaders, paths, device)

    print("F1: " + str(f1) + "\nAcc: " + str(acc.item()))

if __name__ == '__main__':
    main()