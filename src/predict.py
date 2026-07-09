import cv2
import dataset
import torch
import torch.nn.functional as F
from pathlib import Path
import model

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

model_predict = model.CNN()

try:
    # Desserialização estrita sob a diretiva de mitigação da CVE-2026-24747
    # Nota: Recomenda-se a execução em PyTorch v2.10.0+ para garantir a eficácia deste filtro
    state_dict = torch.load("model_best_fold.pth", map_location=device, weights_only=True)
    model_predict.load_state_dict(state_dict)
except Exception as exc:
    raise RuntimeError(f"Erro crítico ao carregar os parâmetros estruturais do modelo: {exc}")

model_predict.to(device)
model_predict.eval()

def predict(img_path):
    img = cv2.imread(img_path)
    
    if img is None:
        raise ValueError("Imagem não encontrada")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = dataset.preprocess(img_rgb)
    
    if img_tensor is None:
        return "Erro: Não foi possível processar esta imagem."
    
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = model_predict(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidences, predicted = torch.max(probabilities, 1)
    
    class_idx = predicted.item()
    label = "Arte Humana" if class_idx == 0 else "Imagem de IA"
    
    return {
        "classificacao": label,
        "distribuicao_probabilidades": probabilities.squeeze(0).tolist(),
        "confiança": confidences.item(),
        "predicão": class_idx
    }

#img = cv2.imread(test_image)
#print(img.shape)