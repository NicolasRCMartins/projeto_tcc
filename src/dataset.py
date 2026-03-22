import os
import cv2
import numpy as np
import torch

from skimage import exposure
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ArtDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        classes = {"humano":0, "ia":1}

        for label_name in classes:
            folder = os.path.join(root_dir, label_name)

            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                self.images.append(path)
                self.labels.append(classes[label_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]

        return img, label

def preprocess(image):

    image = cv2.resize(image, (224,224))

    image = image / 255.0

    image = np.transpose(image, (2,0,1))

    image = np.ascontiguousarray(image)

    image = torch.from_numpy(image).float()

    return image

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = ArtDataset("dataset", transform=transform)

""""
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2 #paralelismo
)
"""

"""
for path in dataset.images:

    img = cv2.imread(path)

    cv2.imshow("Dataset Image", img)

    cv2.waitKey(0)  # espera tecla para ir para próxima

cv2.destroyAllWindows()
"""