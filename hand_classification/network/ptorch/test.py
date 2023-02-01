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
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('-p', '--patience', type=int, default=None, help='Training patience')


    args = parser.parse_args()


    print("Script's arguments: ",args)
    dataset = "test_kinect"
    data_dir = f'{os.getenv("HOME")}/Datasets/ASL/{dataset}'

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Training device: ", device)

    model = InceptioV3_frozen(4, 1)
    model.name = f"{model.name}"
    trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{model.name}/{model.name}.pth', map_location=torch.device('cpu'))
    model.load_state_dict(trained_weights)

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])

    data_transforms = {
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
                    for x in ['test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                shuffle=True, num_workers=2, prefetch_factor=1)
                        for x in ['test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
    class_names = image_datasets['test'].classes

    print("Test classes: ",class_names)

    data_saving_path = os.path.join(data_dir, "results", f"{model.name}", dataset)
    test_path = os.path.join(data_saving_path, f"test_results_{model.name}.csv")
    
    if not os.path.exists(data_saving_path):
        os.mkdir(data_saving_path)

    model.to(device)

    model.eval()
    test_labels = []
    test_preds = []
    running_corrects = 0

    count_tested = 0

    for inputs, labels in dataloaders["test"]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        print("Tested ", count_tested, " out of ", dataset_sizes["test"])
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)
        
        for i in range(0, len(labels)):

            test_labels.append(labels[i].item())
            test_preds.append(preds[i].item())
            count_tested += 1

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