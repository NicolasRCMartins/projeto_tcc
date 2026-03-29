import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)

class CNN(nn.Module): #CNN -> Convolutional Neural Networks

    def __init__(self):
        super(CNN, self).__init__() #inicializa a classe base que delega chamadas de métodos para o tipo de classe de parent ou sibling, permitindo que o Pytorch registre as camadas e parâmetros.

        self.conv1 = nn.Conv2d(3,16,3) #aplica uma convolução 2D em um sinal de input composto de outros muitos "input planes". Resulta em um output de matriz transformado, capturando "patterns" como arestas, texturas e formatos (edges, textures and shapes)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,3) #aumenta a profundidade
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64,3)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2,2) #reduz o tamanho da imagem e mantém os valores mais importantes
        self.adapt = nn.AdaptiveAvgPool2d((7,7))

        self.fc1 = nn.LazyLinear(128) #classificador final, pegando as features extraídas e decidem a classe
        self.fc2 = nn.Linear(128,2)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self,x): #execução real da rede neural
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) #introduz não-linearidade
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = self.adapt(x)

        x = x.view(x.size(0), -1) #transforma e simplifica um array. Ex: [batch, canais, altura, largura] -> [batch, vetor]

        x = F.relu(self.fc1(x)) #reduz dimensionalidade e aprende combinações de features
        x = self.dropout(x)
        x = self.fc2(x) #saída final das classificações

        return x