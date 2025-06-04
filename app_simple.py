from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import json
from datetime import datetime
import random

app = Flask(__name__)
CORS(app)

# Configuración básica
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_emotion_simple():
    """
    Análisis emocional simulado (sin procesamiento real de audio)
    """
    emotions = ['Alegría', 'Tristeza', 'Ansiedad', 'Neutral', 'Enojo']
    emotion = random.choice(emotions)
    confidence = random.randint(70, 95)
    
    return {
        'emotion': emotion,
        'confidence': confidence
    }

# HTML Template simplificado
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Asistente de Apoyo Emocional (Demo)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 1.1em; }
        .content { padding: 40px; }
        .demo-notice {
            background: #fff3cd;
            border: 2px solid #ffeaa7;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            color: #856404;
            text-align: center;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }
        .upload-area:hover { border-color: #764ba2; background: #f8f9ff; }
        .upload-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        .upload-btn:hover { transform: translateY(-2px); }
        .analyze-btn {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 25px;
            font-size: 1.2em;
            cursor: pointer;
            width: 100%;
            margin: 20px 0;
            transition: all 0.3s ease;
        }
        .analyze-btn:hover { transform: translateY(-2px); }
        .result {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 30px;
            border-radius: 15px;
            margin-top: 20px;
            text-align: center;
        }
        .emotion {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .confidence {
            font-size: 1.3em;
            margin-bottom: 20px;
            color: #666;
        }
        .message {
            font-size: 1.1em;
            line-height: 1.6;
            color: #444;
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top: 15px;
        }
        .disclaimer {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            color: #856404;
        }
        .file-info {
            background: #e8f5e8;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            color: #2d5a2d;
        }
        .instructions {
            background: #e3f2fd;
            border: 1px solid #90caf9;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            color: #1565c0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Asistente de Apoyo Emocional</h1>
            <p>Demo - Análisis emocional simulado</p>
        </div>
        
        <div class="content">
            <div class="demo-notice">
                <strong>📋 MODO DEMO:</strong> Esta es una versión de demostración que simula el análisis emocional. 
                Para análisis real de audio, instala las dependencias completas usando permisos de administrador.
            </div>
            
            <div class="instructions">
                <h3>🔧 Para habilitar análisis real de audio:</h3>
                <ol>
                    <li>Ejecuta PowerShell como administrador</li>
                    <li>Navega a esta carpeta</li>
                    <li>Ejecuta: <code>pip install --user librosa soundfile</code></li>
                    <li>O usa: <code>install_admin.bat</code> como administrador</li>
                </ol>
            </div>
            
            <div class="upload-area" onclick="document.getElementById('audioFile').click()">
                <h3>📁 Selecciona tu archivo de audio</h3>
                <p>Formatos soportados: WAV, MP3, FLAC, M4A</p>
                <br>
                <button class="upload-btn" type="button">Elegir Archivo</button>
                <input type="file" id="audioFile" accept=".wav,.mp3,.flac,.m4a" onchange="handleFileSelect(this)" style="display: none;">
            </div>
            
            <div id="fileInfo" class="file-info" style="display: none;"></div>
            
            <button id="analyzeBtn" class="analyze-btn" onclick="analyzeAudio()">
                🔍 Analizar Emoción (Demo)
            </button>
            
            <div id="result" style="display: none;"></div>
            
            <div class="disclaimer">
                <strong>⚠️ Importante:</strong> Este sistema es solo un apoyo emocional y no sustituye el diagnóstico profesional. 
                Consulta siempre a un especialista en salud mental para obtener ayuda profesional.
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;
        
        function handleFileSelect(input) {
            const file = input.files[0];
            if (file) {
                selectedFile = file;
                document.getElementById('fileInfo').style.display = 'block';
                document.getElementById('fileInfo').innerHTML = `
                    <strong>📄 Archivo seleccionado:</strong> ${file.name}<br>
                    <strong>📏 Tamaño:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB<br>
                    <strong>🎵 Tipo:</strong> ${file.type}
                `;
                document.getElementById('result').style.display = 'none';
            }
        }
        
        async function analyzeAudio() {
            if (!selectedFile) {
                alert('Por favor selecciona un archivo de audio primero');
                return;
            }
            
            // Simular análisis
            document.getElementById('analyzeBtn').innerHTML = '🔄 Analizando...';
            document.getElementById('analyzeBtn').disabled = true;
            
            setTimeout(async () => {
                try {
                    const response = await fetch('/analyze_demo', {
                        method: 'POST'
                    });
                    
                    const data = await response.json();
                    showResult(data);
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    document.getElementById('analyzeBtn').innerHTML = '🔍 Analizar Emoción (Demo)';
                    document.getElementById('analyzeBtn').disabled = false;
                }
            }, 2000);
        }
        
        function showResult(data) {
            const messages = {
                'Alegría': '🌟 ¡Qué maravilloso! Tu energía positiva es contagiosa. Sigue cultivando esos momentos de felicidad.',
                'Tristeza': '💙 Es normal sentir tristeza. Estos sentimientos son temporales. No dudes en buscar apoyo.',
                'Ansiedad': '🌸 Respira profundo. La ansiedad puede ser abrumadora, pero tienes la fuerza para superarla.',
                'Neutral': '⚖️ Un estado equilibrado es valioso. Aprovecha esta calma para reflexionar.',
                'Enojo': '🔥 El enojo es válido. Trata de identificar la causa y busca formas constructivas de expresarlo.'
            };
            
            const emotionEmojis = {
                'Alegría': '😊',
                'Tristeza': '😢',
                'Ansiedad': '😰',
                'Neutral': '😐',
                'Enojo': '😠'
            };
            
            document.getElementById('result').innerHTML = `
                <div class="result">
                    <div class="emotion">${emotionEmojis[data.emotion]} ${data.emotion}</div>
                    <div class="confidence">Confianza: ${data.confidence}% (Simulado)</div>
                    <div class="message">${messages[data.emotion]}</div>
                </div>
            `;
            document.getElementById('result').style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze_demo', methods=['POST'])
def analyze_demo():
    """Endpoint de demostración"""
    result = analyze_emotion_simple()
    return jsonify({
        'emotion': result['emotion'],
        'confidence': result['confidence'],
        'timestamp': datetime.now().isoformat(),
        'mode': 'demo'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy - demo mode'})

if __name__ == '__main__':
    print("🧠 Asistente de Apoyo Emocional - MODO DEMO")
    print("=" * 50)
    print("🌐 Servidor: http://localhost:5000")
    print("⚠️  MODO DEMO: Análisis simulado")
    print("📋 Para análisis real, instala librosa con permisos de admin")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
