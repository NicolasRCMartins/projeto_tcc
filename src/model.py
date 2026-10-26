import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        weights = ResNet34_Weights.DEFAULT
        self.base_model = resnet34(weights=weights)
        
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.base_model(x)