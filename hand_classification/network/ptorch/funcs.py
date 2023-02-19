import torch
from networks import InceptioV3_frozen, InceptioV3_unfrozen, InceptionV3
import os
import time
from torchvision import datasets, transforms
import numpy as np
import csv
import copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, \
    precision_score, f1_score

def choose_model(model_name, device):

    if model_name.lower() == "inceptionv3_unfrozen":

        model = InceptioV3_unfrozen(4, 1)

        print(f"Using {model_name}")

        model.name = model_name
        trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{model.name}/{model.name}.pth', map_location=torch.device(device))
        model.load_state_dict(trained_weights)
        
        return model

    if model_name.lower() == "inceptionv3_frozen":

        model = InceptioV3_frozen(4, 1)

        print(f"Using {model_name}")

        model.name = model_name
        trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{model.name}/{model.name}.pth', map_location=torch.device(device))
        model.load_state_dict(trained_weights)
        
        return model
    

    if model_name.lower() == "inceptionv3_frozen_aug":

        model = InceptioV3_frozen(4, 1)

        print(f"Using {model_name}")

        model.name = model_name
        trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{model.name}/{model.name}.pth', map_location=torch.device(device))
        model.load_state_dict(trained_weights)
        
        return model
    


    if model_name.lower() == "inceptionv3_unfrozen_aug":

        model = InceptioV3_unfrozen(4, 1)

        print(f"Using {model_name}")

        model.name = model_name
        trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{model.name}/{model.name}.pth', map_location=torch.device(device))
        model.load_state_dict(trained_weights)
        
        return model

    if  "inceptionv3" in model_name.lower():
        
        model = InceptionV3(4, 1, device)
        model.name = model_name
        print(f"Using {model_name}")

        model.name = model_name
        trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{model.name}/{model.name}.pth', map_location=torch.device(device))
        model.load_state_dict(trained_weights)
        
        return model


def train(model, dataloaders,  paths, args, device):

    if not os.path.exists(os.path.join(paths["results"], f"{model.name}")):
        os.mkdir(os.path.join(paths["results"], f"{model.name}"))

    since = time.time()

    best_loss = 0.0

    history_collumns = ["epoch", "train_loss", "val_loss", 'train_con_loss', 'val_con_loss', 'train_acc', 'val_acc']
    history = {'train_loss': [], 'val_loss': [], 'train_con_loss': [], 'val_con_loss': [], 'train_acc': [], 'val_acc': []}
    early_stopping_counter = 0
    last_epoch_loss = np.inf
    early_stopping = False


    for epoch in range(args.epochs):
        # if (epoch + 1) % 10 == 0:
        #     print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        #     print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_con_loss = 0.0
            running_corrects = 0
            
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

            epoch_loss = running_loss / dataloaders["dataset_sizes"][phase]
            epoch_con_loss = running_con_loss / dataloaders["dataset_sizes"][phase]
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_con_loss"].append(epoch_con_loss)

            epoch_acc = running_corrects.double() / dataloaders["dataset_sizes"][phase]
            history[f"{phase}_acc"].append(epoch_acc.item())


            # deep copy the model

            if phase == 'train' and epoch_loss < best_loss:
                best_loss = epoch_loss

            if phase == 'val' and epoch_loss > last_epoch_loss:
                early_stopping_counter += 1
                # print("Counter: ", early_stopping_counter)
                if early_stopping_counter >= args.patience:
                    # print('Early stopping!')
                    early_stopping = True
                    break

            elif phase == 'val' and epoch_loss <= last_epoch_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                # print("Reset counter")
                early_stopping_counter = 0

            if phase == 'val':
                last_epoch_loss = epoch_loss
                
            # if (epoch + 1) % 10 == 0:
            #     print('{} Loss: {:.4f} - Con Loss: {:.4f} - Acc: {:.4f}'.format(
            #         phase, epoch_loss, epoch_con_loss, epoch_acc))


            # print("Checkpoint - Saving training data")
            # torch.save(best_model_wts, os.path.join(paths["results"], f"{model.name}", f"{model.name}.pth")) 

            # print("Data saved in : ", os.path.join(paths["results"], f"{model.name}"))
            # with open(os.path.join(paths["results"], f"{model.name}", f"{model.name}_train.csv"), 'w') as csvfile:
            #     writer = csv.DictWriter(csvfile, fieldnames=history_collumns)
            #     writer.writeheader()
            #     for i in range(0, len(history["train_loss"])):

            #         row = {"epoch": i+1, 
            #             "train_loss": history[f"train_loss"][i], 
            #             "val_loss": history[f"val_loss"][i],
            #             "train_con_loss": history[f"train_con_loss"][i],
            #             "val_con_loss": history[f"val_con_loss"][i],
            #             "train_acc": history[f"train_acc"][i],
            #             "val_acc": history[f"val_acc"][i]}
                    
            #         writer.writerow(row)

        if early_stopping is True:
            break
    
                
    del inputs, labels, contrastive_features, outputs, preds
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:8f}'.format(best_loss))

    return best_model_wts


def test(model, dataloaders, paths, device):

    if not os.path.exists(os.path.join(paths["results"], f'{model.name}', "multi_user")):
        os.mkdir(os.path.join(paths["results"], f'{model.name}', "multi_user"))

    model.eval()
    test_labels = []
    test_preds = []
    logits = []
    file_names = []
    running_corrects = 0

    count_tested = 0

    for j, (inputs, labels) in enumerate(dataloaders["test"], 0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            
            for i in range(0, len(labels)):

                idx = dataloaders["test"].batch_size * j + i

                sample_fname, _ = dataloaders.dataset.samples[idx]

                test_labels.append(labels[i].item())
                test_preds.append(preds[i].item())
                file_names.append(sample_fname)

                logit = outputs[i]

                # logit = softmax(logit)

                logits.append(logit.to('cpu').detach().numpy())
                count_tested += 1

    with open(os.path.join(paths["results"], f'{model.name}', "multi_user", f'{model.name}_test.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["labels", "predictions", "logits", "filename"])
        writer.writeheader()
        for i in range(0, len(test_preds)):

            row = {"labels": test_labels[i], 
                    "predictions": test_preds[i],
                    "logits": logits[i],
                    "filename": file_names[i]}
            
            writer.writerow(row)


    test_acc = running_corrects.double() / dataloaders["dataset_sizes"]["test"]

    # print('Test Accuracy of the model on the {:.0f} test images: {:.2f}'.format(dataloaders["dataset_sizes"]["test"], 100 * test_acc))

    f1 = f1_score(test_labels, test_preds, average=None)

    return np.mean(f1), test_acc

def set_transforms(args, s, p):

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(299),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    if args.augmentation:
        data_transforms['train'] = transforms.Compose([
            transforms.Resize(299),
            transforms.RandomApply(torch.nn.ModuleList([transforms.RandomResizedCrop(299, scale=(1-s, 1+s))]), p=p),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return data_transforms