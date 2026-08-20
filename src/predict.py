import cv2
import dataset
import torch
import torch.nn.functional as F
from pathlib import Path
import model
import os
import uuid
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

model_predict = model.CNN()

try:
    state_dict = torch.load("model_best_fold.pth", map_location=device, weights_only=True)
    model_predict.load_state_dict(state_dict)
except Exception as exc:
    raise RuntimeError(f"Erro crítico ao carregar os parâmetros estruturais do modelo: {exc}")

model_predict.to(device)
model_predict.eval()

def predict(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"erro": "Não foi possível processar esta imagem. Verifique se é um arquivo válido."}
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = dataset.preprocess(img_rgb)
    
    if img_tensor is None:
        return {"erro": "Erro no pré-processamento da imagem."}
    
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = model_predict(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        prob_humano = probabilities[0][0].item()
        prob_ia = probabilities[0][1].item()
        
        threshold = 0.8
        
        if prob_ia >= threshold:
            class_idx = 1  
            label = "Imagem de IA"
            confidence = prob_ia  
        else:
            class_idx = 0  
            label = "Arte Humana"
            confidence = prob_humano  
    
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join('static', 'uploads', filename)
    cv2.imwrite(filepath, img)
    
    return {
        "classificacao": label,
        "confianca": f"{confidence * 100:.2f}%",
        "predicao": class_idx,
        "imagem_url": f"/{filepath}".replace("\\", "/"),
    }