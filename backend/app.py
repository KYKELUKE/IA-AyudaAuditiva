from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import librosa
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuración
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a'}
UPLOAD_FOLDER = 'uploads'

# Crear carpeta de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_features(file_path):
    """Extrae características del audio"""
    try:
        print(f"🎵 Procesando archivo: {file_path}")
        
        # Cargar audio
        y, sr = librosa.load(file_path, sr=22050, duration=30)
        
        if len(y) == 0:
            print("❌ Archivo de audio vacío")
            return None
        
        features = {}
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = float(np.mean(mfccs))
        features['mfcc_std'] = float(np.std(mfccs))
        
        # Chroma
        chroma = librosa.feature.chroma(y=y, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        
        # Características espectrales
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid'] = float(np.mean(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr'] = float(np.mean(zcr))
        
        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
        except:
            features['tempo'] = 120.0  # Valor por defecto
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features['rms'] = float(np.mean(rms))
        
        print(f"✅ Características extraídas: {len(features)} features")
        return features
        
    except Exception as e:
        print(f"❌ Error extrayendo características: {e}")
        return None

def analyze_emotion_simple(features):
    """Análisis emocional basado en reglas simples"""
    if not features:
        return None
    
    # Obtener características
    spectral_centroid = features.get('spectral_centroid', 1500)
    tempo = features.get('tempo', 120)
    zcr = features.get('zcr', 0.05)
    rms = features.get('rms', 0.1)
    mfcc_mean = features.get('mfcc_mean', 0)
    
    print(f"🔍 Analizando: tempo={tempo:.1f}, spectral={spectral_centroid:.1f}, zcr={zcr:.3f}")
    
    # Lógica de clasificación mejorada
    if spectral_centroid > 2000 and tempo > 120 and rms > 0.15:
        emotion = 'Alegría'
        confidence = min(85 + int((spectral_centroid - 2000) / 100), 95)
    elif spectral_centroid < 1200 and tempo < 90 and rms < 0.1:
        emotion = 'Tristeza'
        confidence = min(80 + int((1200 - spectral_centroid) / 50), 92)
    elif zcr > 0.08 and tempo > 130:
        emotion = 'Ansiedad'
        confidence = min(75 + int(zcr * 1000), 88)
    elif tempo > 140 and rms > 0.2:
        emotion = 'Enojo'
        confidence = min(78 + int((tempo - 140) / 5), 90)
    else:
        emotion = 'Neutral'
        confidence = 70 + int(abs(mfcc_mean) * 10)
    
    return {
        'emotion': emotion,
        'confidence': max(65, min(confidence, 95)),  # Entre 65% y 95%
        'features': features
    }

# HTML template para la interfaz web
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Asistente de Apoyo Emocional</title>
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
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }
        .upload-area:hover { border-color: #764ba2; background: #f8f9ff; }
        .upload-area input[type="file"] {
            display: none;
        }
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
        .analyze-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
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
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .file-info {
            background: #e8f5e8;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            color: #2d5a2d;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Asistente de Apoyo Emocional</h1>
            <p>Análisis emocional inteligente a través de tu voz</p>
        </div>
        
        <div class="content">
            <div class="upload-area" onclick="document.getElementById('audioFile').click()">
                <h3>📁 Selecciona tu archivo de audio</h3>
                <p>Formatos soportados: WAV, MP3, FLAC, M4A</p>
                <br>
                <button class="upload-btn" type="button">Elegir Archivo</button>
                <input type="file" id="audioFile" accept=".wav,.mp3,.flac,.m4a" onchange="handleFileSelect(this)">
            </div>
            
            <div id="fileInfo" class="file-info" style="display: none;"></div>
            
            <button id="analyzeBtn" class="analyze-btn" onclick="analyzeAudio()" disabled>
                🔍 Analizar Emoción
            </button>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>Analizando tu audio... Esto puede tomar unos segundos</p>
            </div>
            
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
                document.getElementById('analyzeBtn').disabled = false;
                document.getElementById('result').style.display = 'none';
            }
        }
        
        async function analyzeAudio() {
            if (!selectedFile) {
                alert('Por favor selecciona un archivo de audio primero');
                return;
            }
            
            const formData = new FormData();
            formData.append('audio', selectedFile);
            
            // Mostrar loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('analyzeBtn').disabled = true;
            document.getElementById('result').style.display = 'none';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showResult(data);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error de conexión: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = false;
            }
        }
        
        function showResult(data) {
            const messages = {
                'Alegría': '🌟 ¡Qué maravilloso! Tu energía positiva es contagiosa. Sigue cultivando esos momentos de felicidad y compártelos con otros.',
                'Tristeza': '💙 Es completamente normal sentir tristeza. Estos sentimientos son temporales y forman parte de la experiencia humana. No dudes en buscar apoyo.',
                'Ansiedad': '🌸 Respira profundo y despacio. La ansiedad puede ser abrumadora, pero recuerda que tienes la fuerza para superarla. Considera técnicas de relajación.',
                'Neutral': '⚖️ Un estado equilibrado es muy valioso. Aprovecha esta calma para reflexionar, planificar y cuidar tu bienestar mental.',
                'Enojo': '🔥 El enojo es una emoción válida que nos indica que algo necesita atención. Trata de identificar la causa y busca formas constructivas de expresarlo.'
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
                    <div class="confidence">Confianza: ${data.confidence}%</div>
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

@app.route('/api')
def api_info():
    return jsonify({
        'message': 'Asistente de Apoyo Emocional API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'Interfaz web',
            '/analyze': 'Análisis de audio (POST)',
            '/health': 'Estado del servicio'
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    try:
        print("🎯 Nueva solicitud de análisis recibida")
        
        if 'audio' not in request.files:
            return jsonify({'error': 'No se encontró archivo de audio'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Formato de archivo no permitido. Use WAV, MP3, FLAC o M4A'}), 400
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
        
        try:
            file.save(temp_path)
            print(f"📁 Archivo guardado temporalmente: {temp_path}")
            
            # Extraer características
            features = extract_audio_features(temp_path)
            
            if features is None:
                return jsonify({'error': 'Error al procesar el archivo de audio. Verifique que sea un archivo válido.'}), 500
            
            # Analizar emoción
            result = analyze_emotion_simple(features)
            
            if result is None:
                return jsonify({'error': 'Error en el análisis emocional'}), 500
            
            print(f"✅ Análisis completado: {result['emotion']} ({result['confidence']}%)")
            
            response = {
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'timestamp': datetime.now().isoformat(),
                'filename': filename
            }
            
            return jsonify(response)
        
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                print(f"🗑️ Archivo temporal eliminado: {temp_path}")
    
    except Exception as e:
        print(f"❌ Error en análisis: {str(e)}")
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

if __name__ == '__main__':
    print("🧠 Iniciando Asistente de Apoyo Emocional")
    print("=" * 50)
    print("🌐 Servidor: http://localhost:5000")
    print("📱 Interfaz web: http://localhost:5000")
    print("🔗 API: http://localhost:5000/api")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
