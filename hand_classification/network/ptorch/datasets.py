import os
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(data_dir,
                           batch_size,
                           transfroms,
                           random_seed,
                           split=[0.2, 0.3],
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((split[0] + split[1] >= 0) and (split[0] + split[1] <= 1)), error_msg

    # load the dataset
    train_dataset = datasets.ImageFolder(
        root=data_dir, transform=transfroms["train"],
    )

    valid_dataset = datasets.ImageFolder(
        root=data_dir, transform=transfroms["val"],
    )

    test_dataset = datasets.ImageFolder(
        root=data_dir, transform=transfroms["test"],
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    test_split = int(np.floor(split[1] * num_train))
    val_split = int(np.floor(split[0] * num_train)) + test_split


    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    test_idx, val_idx, train_idx = indices[0:test_split], indices[test_split:val_split], indices[val_split:]
    print(f"Number of dataset images: {num_train} ({round(num_train * 100 / len(train_dataset), 2)} %)")
    print(f"Number of train images: {len(train_idx)} (({round(len(train_idx) * 100 / len(train_dataset), 2)} %)")
    print(f"Number of val images: {len(val_idx)} ({round(len(val_idx) * 100 / len(train_dataset), 2)} %)")
    print(f"Number of test images: {len(test_idx)} ({round(len(test_idx) * 100 / len(train_dataset), 2)} %)")
    print(f"Number of total images: {len(test_idx) + len(train_idx) + len(val_idx)}")

    class_names = train_dataset.classes

    print("Training classes: ",class_names)

    # train_idx, valid_idx = indices[test_split:], indices[:test_split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    dataset_sizes = {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)}

    return (train_loader, valid_loader, test_loader, dataset_sizes)

if __name__ == "__main__":

    data_dir = f'{os.getenv("HOME")}/Datasets/ASL/kinect'

    
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



    get_train_valid_loader(data_dir, 64, data_transforms, None, shuffle=True)