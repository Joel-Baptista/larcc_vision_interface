import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from hand_gesture_recognition.utils.losses import SupConLoss

class InceptionV3(nn.Module):
    def __init__(self, num_classes, learning_rate, device, temp= 0.07,unfreeze_layers = [], train_encoder = True,
    con_features = 32, class_features = 256, dropout = 0.0):
        super(InceptionV3, self).__init__()

        self.name = "InceptionV3"
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        
        self.loss = nn.CrossEntropyLoss()
        self.con_loss = SupConLoss(temperature=temp, device=device)
        self.model.aux_logits = False
        self.unfreeze_layers = unfreeze_layers
        self.con_features = con_features
        self.class_features = class_features
        self.drop_out = dropout
        self.train_encoder = train_encoder
        self.learning_rate = learning_rate


        for parameter in self.model.parameters():
            parameter.requires_grad = False

        
        self.model.dropout.p = dropout
        self.model.fc = nn.Identity()

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

        self.proj_head.requires_grad_= True
        # summary(self.proj_head, (2048,))
        print(self.proj_head)

        # self.classifier = nn.Linear(class_features, num_classes)
        self.classifier =  nn.Sequential(
                nn.Linear(class_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 4),
                )
        self.classifier.requires_grad_= True
        print(self.classifier)
        # summary(self.classifier, (2048,))
        cnt = 0
        for child in self.model.children():
            cnt += 1
            if cnt in unfreeze_layers: 
                print(child)
                print("----------------")
                for param in child.parameters():
                    param.requires_grad = True
        
        if len(unfreeze_layers) == 0:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(self.proj_head.parameters())), lr=learning_rate)
            self.optimizer.add_param_group({"params": filter(lambda p: p.requires_grad, list(self.classifier.parameters())), "lr": learning_rate})
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(self.model.parameters())), lr=learning_rate) 
            self.optimizer.add_param_group({"params": filter(lambda p: p.requires_grad, list(self.proj_head.parameters())), "lr": learning_rate})
            self.optimizer.add_param_group({"params": filter(lambda p: p.requires_grad, list(self.classifier.parameters())), "lr": learning_rate})
            

    # def save_outputs_hook(self, layer_id: str):
    #     def fn(_, __, output):
    #         self._features[layer_id] = output
    #     return fn

    def forward(self, x):

        features = self.model(x)

        contrastive_features = self.proj_head(features)
        classes = self.classifier(features)

        # no activation and no softmax at the end
        return classes, contrastive_features

