import os #biblioteca para abrir e manipular diretórios do windows
import cv2 #opencv - open source biblioteca com funções de computação visual em tempo-real, usado principalmente para processamento de vídeo e imagem.
import numpy as np #biblioteca para cálculos computacionais mais precisos e científicos
import torch #pytorch - framework de machine learning para classificação de imagens
from PIL import Image
from skimage.feature import local_binary_pattern

def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()  # verifica corrupção
        return True
    except:
        return False

from torch.utils.data import Dataset # Dataset -> Serve para processamento de amostra de dados que utiliza datasets pré-processados ou "pré-prontos" juntamente com o próprio dataset de um projeto. Armazena os samples e seus respectivos labels.
from torchvision import transforms #transformação de imagens inseridos no dataset. 

class ArtDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Parâmetros do LBP
        self.radius = 1
        self.n_points = 8 * self.radius

        classes = {"humano":0, "ia":1}

        for label_name in classes:
            folder = os.path.join(root_dir, label_name)

            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                
                self.images.append(path)
                self.labels.append(classes[label_name]) #esse bloco de código serve para dar label nas imagens de "humano/ia" em "dataset"

    def __len__(self):
        return len(self.images)
    
    def extract_lbp(self, image):

        # Converte para grayscale
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        if len(image.shape) == 3 and image.shape[0] == 3:
            # provavelmente está em CHW (ERRADO para OpenCV)
            image = np.transpose(image, (1,2,0))
        
        if image.shape[-1] != 3:
            raise ValueError(f"Formato inválido: {image.shape}")
        
        #print(type(image))
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray = gray.astype(np.uint8)

        # Extrai LBP
        lbp = local_binary_pattern(
            gray,
            self.n_points,
            self.radius,
            method='uniform'
        )

        # Normaliza para 0-255
        lbp = (lbp / lbp.max()) * 255

        return lbp

    def __getitem__(self, idx):

        img_path = self.images[idx]
        
        if not is_valid_image(img_path):
            print("Imagem corrompida:", img_path)
            return None
        else:
            img = cv2.imread(img_path) #carrega uma imagem de um path específico
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #converte uma imagem de uma cor para outra, nesse caso converte de BGR (Blue, Green, Red) para RGB (Red, Green, Blue)

            if self.transform:
                 img = self.transform(img)

            label = self.labels[idx]
        
            # ========= LBP =========
            lbp_img = self.extract_lbp(img)
            
            lbp_img = lbp_img.astype(np.float32)

            # Normalização
            lbp_img = lbp_img / 255.0

            # Tensor
            lbp_img = torch.tensor(
                lbp_img,
                dtype=torch.float32
            )
            
            lbp_img = lbp_img.unsqueeze(0)

            label = torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )

            return lbp_img, label

def preprocess(image):
    try:
        #image = image / 255.0 #para melhorar na estabilidade numérica no treinamento

        #image = np.transpose(image, (2,0,1)) #transposição de elementos de matrizes. Ex: [1,2], [3,4] -> [1,3],[2,4]. Isso altera para N matrizes. Serve para reorganizar os eixos para a leitura do Pytorch [channels, height, width]

        #image = np.ascontiguousarray(image) #cria uma cópia do array reorganizado na memória de maneira contígua, significando que elementos de mesma linha do array são adjacentes aos outros na memória. Melhora no desempenho (cache CPU), compatibilidae com Pytorch e evita bugs.

        #image = torch.from_numpy(image).float() #cria um tensor (matriz multidimensional contendo elementos de um único tipo de dado) a partir de um array (array anterior) do numpy. Nesse caso, transforma os elementos do array em float.
        
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
        #gray = gray.astype(np.uint8)

        lbp = local_binary_pattern(
            gray,
            8,
            1,
            method='uniform'
        )

        lbp = lbp.astype(np.float32)

        lbp = lbp / 255.0

        lbp = torch.tensor(
            lbp,
            dtype=torch.float32
        ).unsqueeze(0)

        return lbp
            
    except:
        print("Erro em:", image)
        
    return image

transform = transforms.Compose([ #para compôr vários tipos de alterações na imagem de uma só vez juntos.
    transforms.ToPILImage(), #converte a imagem para formato PIL (Python Imaging Library)
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(), #Transforma em um tensor
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