import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        weights = ResNet18_Weights.DEFAULT
        self.base_model = resnet18(weights=weights)
        
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.base_model(x)