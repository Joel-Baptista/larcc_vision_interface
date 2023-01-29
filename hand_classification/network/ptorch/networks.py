import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from torchsummary import summary

class InceptioV3_frozen(nn.Module):
    def __init__(self, num_classes):
        super(InceptioV3_frozen, self).__init__()

        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        self.model.aux_logits = False

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 1024),
            nn.Linear(1024, num_classes)
            )

    def forward(self, x):
        out = self.model(x)
        # no activation and no softmax at the end
        return out