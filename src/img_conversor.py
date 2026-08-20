import os
from PIL import Image

input_dir = "imagens_novas_humanas"

output_dir = "imagens_novas_humanas_resized"

target_size = (224, 224)

os.makedirs(output_dir, exist_ok=True)

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".jfif", ".webp")

for filename in os.listdir(input_dir):
    if filename.lower().endswith(valid_extensions):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                img = img.convert("RGB")

                resized_img = img.resize(target_size, Image.LANCZOS)

                resized_img.save(output_path)

        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")