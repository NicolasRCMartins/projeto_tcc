from flask import Flask, request, render_template, send_from_directory
import predict
import os
import glob
import shutil

app = Flask(__name__)

# ==========================================
# 1. LIMPEZA AUTOMÁTICA NO STARTUP
# ==========================================
upload_folder = os.path.join('static', 'uploads')
if os.path.exists(upload_folder):
    files = glob.glob(os.path.join(upload_folder, '*'))
    for f in files:
        os.remove(f)
    print(f"🧹 Pasta de uploads limpa! {len(files)} arquivos antigos removidos.")
else:
    os.makedirs(upload_folder)
    print("📁 Pasta de uploads criada.")
    
# ==========================================
# 2. RESOLUÇÃO DE CAMINHO ABSOLUTO
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'dataset'))
print(f"🔍 O Flask está procurando o dataset em: {DATASET_PATH}")

# ==========================================
# 3. FUNÇÃO DE PÁGINA DE ERRO
# ==========================================
def error_page(message):
    html = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="5;url=/">
        <title>Erro no Processamento</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ background-color: #f8f9fa; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }}
            .error-box {{ text-align: center; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 500px; }}
        </style>
    </head>
    <body>
        <div class="error-box">
            <h1 class="text-danger">⚠️ Ops!</h1>
            <p class="lead">{message}</p>
            <p class="text-muted small">Você será redirecionado automaticamente para a página inicial em 5 segundos...</p>
            <a href="/" class="btn btn-primary mt-3">Voltar agora</a>
        </div>
    </body>
    </html>
    """
    return html, 400 

# ==========================================
# 4. ROTAS DA APLICAÇÃO
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/sobre")
def sobre():
    return render_template('sobre.html')

# AQUI ESTAVA O ERRO: Agora a função que lista as imagens TEM a rota @app.route
@app.route("/dataset")
def show_dataset():
    # 1. Captura a página atual da URL (ex: /dataset?page=2). Padrão é 1.
    page = request.args.get('page', 1, type=int)
    filter_type = request.args.get('filter', 'all')
    per_page = 25
    
    all_images = []
    
    humano_folder = os.path.join(DATASET_PATH, 'humano')
    if os.path.exists(humano_folder):
        for filename in os.listdir(humano_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                all_images.append({
                    'filename': filename,
                    'url': f"/dataset-img/humano/{filename}",
                    'label': 'Humano'
                })
    
    ia_folder = os.path.join(DATASET_PATH, 'ia')
    if os.path.exists(ia_folder):
        for filename in os.listdir(ia_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                all_images.append({
                    'filename': filename,
                    'url': f"/dataset-img/ia/{filename}",
                    'label': 'IA'
                })
                
    if filter_type == 'humano':
        filtered_images = [img for img in all_images if img['label'] == 'Humano']
    elif filter_type == 'ia':
        filtered_images = [img for img in all_images if img['label'] == 'IA']
    else:
        filtered_images = all_images
                
    total_images = len(filtered_images)
    
    total_pages = (total_images + per_page - 1) // per_page if total_images > 0 else 1
    
    if page > total_pages and total_pages > 0:
        page = total_pages
    if page < 1:
        page = 1
        
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated_images = filtered_images[start_index:end_index]
                
    print(f"Filtro: '{filter_type}' | Página {page}/{total_pages} | Mostrando {len(paginated_images)} de {total_images} imagens.")
    
    return render_template(
        'dataset.html', 
        images=paginated_images, 
        total=total_images,
        current_page=page,
        total_pages=total_pages,
        has_prev=page > 1,
        has_next=page < total_pages,
        current_filter=filter_type,
    )

# Rota separada para servir os arquivos de imagem (evita conflito com a página /dataset)
@app.route('/dataset-img/<folder>/<filename>')
def serve_dataset_image(folder, filename):
    if folder not in ['humano', 'ia']:
        return "Acesso negado", 403
    
    directory = os.path.join(DATASET_PATH, folder)
    return send_from_directory(directory, filename)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return error_page("Nenhum arquivo foi enviado no formulário.")
        
    file = request.files['file']
    
    if file.filename == '':
        return error_page("Nenhum arquivo foi selecionado. Por favor, escolha uma imagem.")
        
    if file:
        # Nota: Verifique se a função no seu predict.py se chama 'predict' ou 'predict_image'
        result = predict.predict(file) 
        
        if "erro" in result:
            return error_page(result["erro"])
            
        return render_template('result.html', result=result)
    
@app.route('/feedback', methods=['POST'])
def feedback():
    filepath = request.form.get('filepath')
    correct_label = request.form.get('correct_label') 
    
    src_path = filepath.lstrip('/') 
    
    if not os.path.exists(src_path):
        return "Erro: Imagem não encontrada no servidor.", 404
        
    # CORREÇÃO: Usar DATASET_PATH em vez de '..'
    dest_folder = os.path.join(DATASET_PATH, correct_label)
    os.makedirs(dest_folder, exist_ok=True) 
    
    dest_path = os.path.join(dest_folder, os.path.basename(src_path))
    shutil.move(src_path, dest_path)
    
    return f"""
    <div style="text-align: center; margin-top: 50px; font-family: sans-serif;">
        <h2>✅ Obrigado pelo feedback!</h2>
        <p>A imagem foi movida com sucesso para a pasta de treino (<strong>{correct_label}</strong>).</p>
        <p>Ela será utilizada no próximo retreinamento do modelo para corrigir esse erro.</p>
        <a href="/" style="text-decoration: none; background: #0d6efd; color: white; padding: 10px 20px; border-radius: 5px;">Voltar para o início</a>
    </div>
    """

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)