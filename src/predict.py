import cv2
import dataset
import torch
import model
import torchvision.models as models

model_predict = model.CNN()
model_predict.load_state_dict(torch.load("model.pth"))
model_predict.eval()

def predict(img_path):

    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Imagem não encontrada")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = dataset.preprocess(img)

    img = img.unsqueeze(0) #retorna um novo tensor com uma dimensão de 1

    with torch.no_grad():
        output = model_predict(img)

    pred = torch.argmax(output,1) #retorna os índices com o valor máximo de todos os elementos do tensor

    if pred.item() == 0:
        return "Arte Humana"
    else:
        return "Imagem de IA"

#img = cv2.imread(test_image)
#print(img.shape)