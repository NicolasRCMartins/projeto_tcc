import os
from PIL import Image

# Diretório de entrada (originais)
input_dir = "teste_imagem"

# Diretório de saída (redimensionadas)
output_dir = "teste_imagem_resized"

# Tamanho alvo
target_size = (224, 224)

# Cria a pasta de saída se não existir
os.makedirs(output_dir, exist_ok=True)

# Extensões suportadas
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

# Loop nas imagens
for filename in os.listdir(input_dir):
    if filename.lower().endswith(valid_extensions):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                # Converte para RGB (evita erro com PNGs com alpha, etc)
                img = img.convert("RGB")

                # Redimensiona
                resized_img = img.resize(target_size, Image.LANCZOS)

                # Salva no novo diretório
                resized_img.save(output_path)

        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")