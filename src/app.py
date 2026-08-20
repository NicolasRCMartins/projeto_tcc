from flask import Flask, request, render_template
import predict
import os
import glob
import shutil

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')
if os.path.exists(upload_folder):
    files = glob.glob(os.path.join(upload_folder, '*'))
    for f in files:
        os.remove(f)
    print(f"🧹 Pasta de uploads limpa! {len(files)} arquivos antigos removidos.")
else:
    os.makedirs(upload_folder)
    print("📁 Pasta de uploads criada.")
    
def error_page(message):
    html = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <!-- A MÁGICA ACONTECE AQUI: Redireciona para a raiz (/) em 5 segundos -->
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return error_page("Nenhum arquivo foi enviado no formulário.")
        
    file = request.files['file']
    
    if file.filename == '':
        return error_page("Nenhum arquivo foi selecionado. Por favor, escolha uma imagem.")
        
    if file:
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
        
    dest_folder = os.path.join('..', 'dataset', correct_label)
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