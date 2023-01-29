import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torchsummary import summary

class InceptioV3_frozen(nn.Module):
    def __init__(self, num_classes, learning_rate):
        super(InceptioV3_frozen, self).__init__()

        self.name = "InceptioV3_frozen"
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
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, num_classes)
            )
        self.optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=learning_rate)
    
    def forward(self, x):
        out = self.model(x)
        # no activation and no softmax at the end
        return out