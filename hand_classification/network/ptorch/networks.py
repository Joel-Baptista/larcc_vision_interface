import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torchsummary import summary
from losses import SupConLoss, SimpleConLoss


class InceptionV3(nn.Module):
    def __init__(self, num_classes, learning_rate, device, temp= 0.7,unfreeze_layers = [], con_features = 32, class_features = 256, dropout = 0.0):
        super(InceptionV3, self).__init__()

        self.name = "InceptionV3"
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

        self.loss = nn.CrossEntropyLoss()
        self.con_loss = SupConLoss(temperature=temp, device=device)
        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.model.aux_logits = False

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        
        self.model.dropout.p = dropout
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, class_features),
            nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Linear(128, 32),
            # nn.ReLU(),
            )
        self.model.fc.requires_grad = True


        modules_proj_head = []
        last_output = class_features
        stop = False
        while True:
            new_output = int(last_output / 2)

            if new_output <= con_features:
                new_output = con_features
                stop = True       

            modules_proj_head.append(nn.Linear(last_output, new_output))   
            modules_proj_head.append(nn.ReLU())

            last_output = new_output

            if stop:
                break

        
        self.proj_head = nn.Sequential(
            *modules_proj_head
            )

        # self.proj_head = nn.Sequential(
        #     nn.Linear(class_features, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     )
        self.proj_head.requires_grad_= True


        self.classifier = nn.Linear(class_features, num_classes)
        self.classifier.requires_grad_= True
        # self.model.add_module('head_proj', nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     ))

        # self.model.add_module('classifier', nn.Linear(32, num_classes))
        # self.model.classifier.requires_grad = True


        # self.layers = layers_to_hook
        # self._features = {layer: torch.empty(0) for layer in layers_to_hook}

        # for layer_id in layers_to_hook:
        #     layer = dict([*self.model.named_modules()])[layer_id]
        #     layer.register_forward_hook(self.save_outputs_hook(layer_id))

        cnt = 0
        for child in self.model.children():
            cnt += 1
            if cnt in unfreeze_layers: 
                # print(child)
                # print("----------------")
                for param in child.parameters():
                    param.requires_grad = True
        
        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(self.model.parameters())), lr=learning_rate) # old 

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(self.model.parameters())), lr=learning_rate)
        self.optimizer.add_param_group({"params": filter(lambda p: p.requires_grad, list(self.proj_head.parameters())), "lr": learning_rate})
        self.optimizer.add_param_group({"params": filter(lambda p: p.requires_grad, list(self.classifier.parameters())), "lr": learning_rate})


    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):

        features = self.model(x)

        contrastive_features = self.proj_head(features)
        classes = self.classifier(features)

        # no activation and no softmax at the end
        return classes, contrastive_features



class InceptioV3_frozen(nn.Module):
    def __init__(self, num_classes, learning_rate):
        super(InceptioV3_frozen, self).__init__()

        self.name = "InceptionV3_frozen"
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.model.aux_logits = False

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.dropout.p = 0.0
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
            )


        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)

    def forward(self, x, output_mode = "logits"):

        if output_mode == "features":
            print("self._features")
            print(self._features)
            out = self._features
        else:
            out = self.model(x)

        # no activation and no softmax at the end
        return out

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn


class InceptioV3_unfrozen(nn.Module):
    def __init__(self, num_classes, learning_rate, layers_to_hook = []):
        super(InceptioV3_unfrozen, self).__init__()

        self.name = "InceptionV3_unfrozen"
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.model.aux_logits = False
        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.dropout.p = 0.0
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
            )

        self.layers = layers_to_hook
        self._features = {layer: torch.empty(0) for layer in layers_to_hook}

        for layer_id in layers_to_hook:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))
        
        cnt = 0
        for child in self.model.children():
            cnt += 1
            if cnt == 18: # Last inception block
                print(child)
                print("----------------")
                for param in child.parameters():
                    param.requires_grad = True

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
    
    def forward(self, x, output_mode = "logits"):

        if output_mode == "features":
            print("self._features")
            print(self._features)
            out = self._features
        else:
            out = self.model(x)

        # no activation and no softmax at the end
        return out

    def save_outputs_hook(self, layer_id: str):
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn