from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
import predict
import os
import glob
import shutil
import uuid
from datetime import datetime

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

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
CURADORIA_FOLDER = os.path.join(DATASET_PATH, 'curadoria')
os.makedirs(CURADORIA_FOLDER, exist_ok=True)
print(f"🔍 O Flask está procurando o dataset em: {DATASET_PATH}")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    """Verifica se a extensão do arquivo é permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&display=swap" rel="stylesheet">
        <style>
            body {{ background-color: rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }}
            .error-box {{ text-align: center; background: white; padding: 40px;  box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 500px; }}
        </style>
    </head>
    <body>
        <div class="error-box">
            <h1 class="text-danger">Ops!</h1>
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
    
    filename = secure_filename(filename)
    directory = os.path.join(DATASET_PATH, folder)
    
    if not os.path.isfile(os.path.join(directory, filename)):
        return "Arquivo não encontrado", 404
    
    return send_from_directory(directory, filename)

@app.route('/predict', methods=['POST'])
@limiter.limit("30 per minute")
def predict_route():
    if 'file' not in request.files:
        return error_page("Nenhum arquivo foi enviado no formulário.")
        
    file = request.files['file']
    
    if file.filename == '':
        return error_page("Nenhum arquivo foi selecionado. Por favor, escolha uma imagem.")
    
    if not allowed_file(file.filename):
        return error_page("Formato de arquivo não permitido. Use apenas PNG, JPG, JPEG ou WEBP.")
    
    original_filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
    filepath = os.path.join(upload_folder, unique_filename)
        
    try:
        file.save(filepath)
        print(f"Arquivo salvo com segurança: {unique_filename}")
        
        file.seek(0)
        
        # Faz a predição
        result = predict.predict(file)
        
        if "erro" in result:
            # Remove o arquivo em caso de erro
            if os.path.exists(filepath):
                os.remove(filepath)
            return error_page(result["erro"])
        
        # Adiciona o caminho da imagem ao resultado
        result['imagem_url'] = f"/static/uploads/{unique_filename}"
        result['imagem_path'] = filepath  # Caminho absoluto para o feedback
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        print(f"Erro no processamento: {str(e)}")
        # Remove o arquivo em caso de exceção
        if os.path.exists(filepath):
            os.remove(filepath)
        return error_page("Erro ao processar a imagem. Tente novamente.")
    
@app.route('/feedback', methods=['POST'])
@limiter.limit("30 per minute")
def feedback():
    filepath = request.form.get('filepath')
    correct_label = request.form.get('correct_label') 
    
    if correct_label not in ['humano', 'ia']:
        return error_page("Label inválido.")
    
    if not filepath or '..' in filepath:
        return error_page("Caminho de arquivo inválido.")
    
    src_path = filepath.lstrip('/') 
    
    uploads_abs = os.path.abspath(upload_folder)
    src_abs = os.path.abspath(src_path)
    
    if not src_abs.startswith(uploads_abs):
        return error_page("Acesso negado: arquivo fora da pasta permitida.")
    
    if not os.path.exists(src_path):
        return "Erro: Imagem não encontrada no servidor.", 404
        
    # CORREÇÃO: Usar DATASET_PATH em vez de '..'
    dest_folder = os.path.join(DATASET_PATH, correct_label)
    os.makedirs(dest_folder, exist_ok=True) 
    
    filename = os.path.basename(src_path)
    curadoria_filename = f"{correct_label}_{filename}"
    dest_path = os.path.join(CURADORIA_FOLDER, curadoria_filename)
    
    try:
        shutil.move(src_path, dest_path)
        print(f"Imagem movida para curadoria: {curadoria_filename}")
    
        return f"""
        <!DOCTYPE html>
        <html lang="pt-br">
        <head>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">
        </head>
        <body style="background-color: #f8f9fa; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; font-family: 'Playfair Display', serif;">
            <div style="text-align: center; background: white; padding: 40px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); max-width: 500px;">
                <h2 style="color: #28a745; margin-bottom: 20px;">Obrigado pelo feedback!</h2>
                <p style="color: #495057; margin-bottom: 15px;">A imagem foi enviada com sucesso para a <strong>fila de curadoria</strong>.</p>
                <p style="color: #6c757d; font-size: 0.9rem; margin-bottom: 25px;">Ela será revisada manualmente antes de integrar o dataset oficial para o próximo retreinamento.</p>
                <a href="/" style="text-decoration: none; background: #738fab; color: white; padding: 12px 30px; border-radius: 8px; font-weight: 600; display: inline-block; transition: 0.3s;">Voltar para o início</a>
            </div>
        </body>
        </html>
        """
    except Exception as e:
        print(f"Erro ao mover arquivo: {str(e)}")
        return error_page("Erro ao processar o feedback.")
    
@app.route("/curadoria")
@limiter.limit("30 per hour")  
def curadoria():
    """Página para revisar imagens enviadas pelos usuários"""
    imagens_pendentes = []
    
    if os.path.exists(CURADORIA_FOLDER):
        for filename in sorted(os.listdir(CURADORIA_FOLDER)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                # Extrai o label sugerido do nome do arquivo (ex: "humano_abc123.jpg")
                label_sugerido = filename.split('_')[0] if '_' in filename else 'desconhecido'
                
                imagens_pendentes.append({
                    'filename': filename,
                    'url': f"/curadoria-img/{filename}",
                    'label_sugerido': label_sugerido
                })
    
    return render_template('curadoria.html', imagens=imagens_pendentes, total=len(imagens_pendentes))

@app.route('/curadoria-img/<filename>')
def serve_curadoria_image(filename):
    filename = secure_filename(filename)
    return send_from_directory(CURADORIA_FOLDER, filename)

@app.route('/curadoria/aprovar', methods=['POST'])
@limiter.limit("30 per minute")
def curadoria_aprovar():
    filename = request.form.get('filename')
    label_final = request.form.get('label_final')
    
    if not filename or label_final not in ['humano', 'ia']:
        return jsonify({'error': 'Dados inválidos'}), 400
    
    filename = secure_filename(filename)
    src_path = os.path.join(CURADORIA_FOLDER, filename)
    
    if not os.path.exists(src_path):
        return jsonify({'error': 'Arquivo não encontrado'}), 404
    
    clean_filename = '_'.join(filename.split('_')[1:]) if '_' in filename else filename
    
    dest_folder = os.path.join(DATASET_PATH, label_final)
    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, clean_filename)
    
    try:
        shutil.move(src_path, dest_path)
        print(f"Imagem aprovada e movida para {label_final}: {clean_filename}")
        return jsonify({'success': True, 'message': f'Imagem movida para {label_final}'})
    except Exception as e:
        print(f"Erro ao mover arquivo: {str(e)}")
        return jsonify({'error': 'Erro ao mover arquivo'}), 500

@app.route('/curadoria/rejeitar', methods=['POST'])
@limiter.limit("30 per minute")
def curadoria_rejeitar():
    filename = request.form.get('filename')
    
    if not filename:
        return jsonify({'error': 'Nome do arquivo não fornecido'}), 400
    
    filename = secure_filename(filename)
    filepath = os.path.join(CURADORIA_FOLDER, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Arquivo não encontrado'}), 404
    
    try:
        os.remove(filepath)
        print(f"Imagem rejeitada e removida: {filename}")
        return jsonify({'success': True, 'message': 'Imagem removida'})
    except Exception as e:
        print(f"Erro ao remover arquivo: {str(e)}")
        return jsonify({'error': 'Erro ao remover arquivo'}), 500
    
@app.errorhandler(413)
def request_entity_too_large(error):
    return error_page("Arquivo muito grande. O tamanho máximo permitido é 5MB."), 413

@app.errorhandler(429)
def ratelimit_handler(error):
    return error_page("Muitas requisições. Aguarde um momento e tente novamente."), 429
        
@app.route('/manifest.json')
def manifest():
    return app.send_static_file('manifest.json')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)