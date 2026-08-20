import os 
import cv2 
import numpy as np 
import torch 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()  
        return True
    except Exception:
        return False

class ArtDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        classes = {"humano":0, "ia":1}

        for label_name in classes:
            folder = os.path.join(root_dir, label_name)

            if not os.path.exists(folder):
                print(f"Aviso: Pasta {folder} não encontrada.")
                continue

            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                self.images.append(path)
                self.labels.append(classes[label_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        
        if not is_valid_image(img_path):
            raise ValueError(f"Imagem corrompida ou inválida: {img_path}")
        
        img = cv2.imread(img_path)
        
        if img is None:
            raise ValueError(f"Não foi possível ler a imagem: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
                img = self.transform(img)

        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        return img, label

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

def preprocess(image_input):
    try:
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError("Não foi possível carregar a imagem a partir do caminho.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
        else:
            raise ValueError("Formato de entrada não suportado.")

        tensor_img = inference_transform(img)

        tensor_img = tensor_img.unsqueeze(0)

        return tensor_img
            
    except Exception as e:
        print(f"Erro no preprocessamento: {e}")
        return None
    
if __name__ == '__main__':
    dataset = ArtDataset("dataset", transform=train_transform)
    print(f"Dataset carregado com sucesso! {len(dataset)} imagens encontradas.")