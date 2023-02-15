from funcs import choose_model
from networks import InceptionV3
import torch
import time
import copy
import torchvision
from torchvision import datasets, transforms
import os
import numpy as np
import csv
import argparse
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
                    prog = 'Pytorch Test',
                    description = 'It test a Pytorch model')
    
    parser.add_argument('-d', '--device', type=str, default="cuda:1", help='Decive used for testing')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('-m', '--model_name', type=str, default="InceptionV3_unfrozen", help='Model name')
    parser.add_argument('-t', '--test_dataset', type=str, default="kinect_test", help='Test dataset name')

    args = parser.parse_args()


    print("Script's arguments: ",args)
    dataset = args.test_dataset
    dataset_path = f'{os.getenv("HOME")}/Datasets/ASL/kinect/{dataset}'

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Training device: ", device)

    # model = choose_model(args.model_name, device)
    model = InceptionV3(4, 0.0001, unfreeze_layers= list(np.arange(13, 19)), device=device, con_features=16)
    model.name = args.model_name

    trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{model.name}/{model.name}.pth', map_location=torch.device(device))
    model.load_state_dict(trained_weights)

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.25, 0.25, 0.25])

    data_transforms = transforms.Compose([
            transforms.Resize(299),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    image_datasets = datasets.ImageFolder(dataset_path, data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=args.batch_size,
                                                shuffle=False, num_workers=2, prefetch_factor=1)
    dataset_sizes = len(image_datasets)
    class_names = image_datasets.classes

    print("Test classes: ",class_names)
    print("Test dataset: ", args.test_dataset)
    inputs, classes = next(iter(dataloaders))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # imshow(out)

    data_saving_path = os.path.join(f'{os.getenv("HOME")}/Datasets/ASL/kinect', "results", f"{model.name}", dataset)
    test_path = os.path.join(data_saving_path, f"test_results_{model.name}.csv")
    
    if not os.path.exists(data_saving_path):
        os.mkdir(data_saving_path)

    model.to(device)

    model.eval()
    test_labels = []
    test_preds = []
    logits = []
    file_names = []
    running_corrects = 0

    count_tested = 0

    softmax = torch.nn.Softmax()

    for j, (inputs, labels) in enumerate(dataloaders, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)


        print("Tested ", count_tested, " out of ", dataset_sizes)
        with torch.no_grad():
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)


        for i in range(0, len(labels)):

            idx = args.batch_size * j + i

            sample_fname, _ = dataloaders.dataset.samples[idx]

            test_labels.append(labels[i].item())
            test_preds.append(preds[i].item())
            file_names.append(sample_fname)

            logit = outputs[i]

            # logit = softmax(logit)

            logits.append(logit.to('cpu').detach().numpy())
            count_tested += 1

    with open(test_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["labels", "predictions", "logits", "filename"])
        writer.writeheader()
        for i in range(0, len(test_preds)):

            row = {"labels": test_labels[i], 
                    "predictions": test_preds[i],
                    "logits": logits[i],
                    "filename": file_names[i]}
            
            writer.writerow(row)


    test_acc = running_corrects.double() / dataset_sizes

    print('Test Accuracy of the model on the {:.0f} test images: {:.2f}'.format(dataset_sizes, 100 * test_acc))


if __name__ == '__main__':
    main()