# Usa uma imagem leve do Python
FROM python:3.10-slim

# Define o diretório de trabalho
WORKDIR /app

# Instala dependências do sistema (necessárias para o OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos do projeto para o container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expõe a porta 7860 (padrão do Hugging Face)
EXPOSE 7860

# Comando para iniciar o app (usando o app.py)
# Nota: Se seu arquivo principal tiver outro nome, mude 'app:app'
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]