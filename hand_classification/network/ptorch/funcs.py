import torch
from networks import InceptioV3_frozen, InceptioV3_unfrozen, InceptionV3
import os

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
        
        model = InceptionV3(4, 1)
        model.name = model_name
        print(f"Using {model_name}")

        model.name = model_name
        trained_weights = torch.load(f'{os.getenv("HOME")}/Datasets/ASL/kinect/results/{model.name}/{model.name}.pth', map_location=torch.device(device))
        model.load_state_dict(trained_weights)
        
        return model