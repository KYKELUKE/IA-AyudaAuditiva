from flask import Flask, request, jsonify, render_template_string

import os
import librosa
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import json
from datetime import datetime
import warnings

# Suprimir warnings para una salida más limpia
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuración simple
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_features_simple(file_path):
    """Extrae características de audio de forma simple y robusta"""
    try:
        print(f"🎵 Procesando: {os.path.basename(file_path)}")
        
        # Cargar audio con librosa
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)
        
        if len(y) == 0:
            return None
            
        # Normalizar
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        features = {}
        
        # Características básicas pero efectivas
        try:
            # MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = float(np.mean(mfccs))
            features['mfcc_std'] = float(np.std(mfccs))
        except:
            features['mfcc_mean'] = 0.0
            features['mfcc_std'] = 1.0
        
        try:
            # Centroide espectral
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
        except:
            features['spectral_centroid'] = 1500.0
        
        try:
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr'] = float(np.mean(zcr))
        except:
            features['zcr'] = 0.05
        
        try:
            # RMS Energy
            rms = librosa.feature.rms(y=y)
            features['rms'] = float(np.mean(rms))
        except:
            features['rms'] = 0.1
        
        try:
            # Tempo
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo) if not np.isnan(tempo) else 120.0
        except:
            features['tempo'] = 120.0
        
        # Estadísticas básicas
        features['duration'] = float(len(y) / sr)
        features['audio_std'] = float(np.std(y))
        
        print(f"✅ Características extraídas: {len(features)}")
        return features
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def analyze_emotion_simple(features):
    """Análisis emocional simplificado pero efectivo"""
    if not features:
        return None
    
    # Obtener características
    spectral_centroid = features.get('spectral_centroid', 1500)
    tempo = features.get('tempo', 120)
    zcr = features.get('zcr', 0.05)
    rms = features.get('rms', 0.1)
    duration = features.get('duration', 10)
    
    # Sistema de puntuación simple
    scores = {'Alegría': 0, 'Tristeza': 0, 'Ansiedad': 0, 'Enojo': 0, 'Neutral': 10}
    
    # Análisis basado en tempo
    if tempo > 130:
        scores['Alegría'] += 25
        scores['Ansiedad'] += 15
    elif tempo < 80:
        scores['Tristeza'] += 25
    else:
        scores['Neutral'] += 10
    
    # Análisis basado en frecuencia
    if spectral_centroid > 2000:
        scores['Alegría'] += 20
    elif spectral_centroid < 1200:
        scores['Tristeza'] += 20
    
    # Análisis basado en energía
    if rms > 0.15:
        scores['Enojo'] += 20
        scores['Alegría'] += 10
    elif rms < 0.08:
        scores['Tristeza'] += 15
    
    # Análisis basado en variabilidad
    if zcr > 0.08:
        scores['Ansiedad'] += 20
    
    # Determinar emoción
    emotion = max(scores, key=scores.get)
    confidence = min(90, max(65, scores[emotion] + 40))
    
    return {
        'emotion': emotion,
        'confidence': confidence,
        'features_analyzed': len(features),
        'audio_duration': duration
    }

# HTML simplificado
HTML_SIMPLE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 Asistente Emocional IA</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', sans-serif;
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
        .content { padding: 40px; }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            cursor: pointer;
        }
        .upload-area:hover { border-color: #764ba2; background: #f8f9ff; }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            margin: 10px;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { background: #ccc; cursor: not-allowed; }
        .result {
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            padding: 30px;
            border-radius: 15px;
            margin-top: 20px;
            text-align: center;
        }
        .emotion { font-size: 3em; margin-bottom: 15px; }
        .confidence { font-size: 1.4em; margin-bottom: 20px; color: #666; }
        .message { font-size: 1.1em; background: white; padding: 20px; border-radius: 10px; }
        .loading { display: none; text-align: center; padding: 20px; }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { background: #f8d7da; color: #721c24; padding: 15px; border-radius: 10px; margin: 15px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Asistente Emocional IA</h1>
            <p>Análisis emocional simple y efectivo</p>
        </div>
        
        <div class="content">
            <div class="upload-area" onclick="document.getElementById('audioFile').click()">
                <h3>📁 Sube tu archivo de audio</h3>
                <p>WAV, MP3, FLAC, M4A, OGG, WEBM</p>
                <br>
                <button class="btn" type="button">📁 Elegir Archivo</button>
                <input type="file" id="audioFile" accept=".wav,.mp3,.flac,.m4a,.ogg,.webm" style="display: none;">
            </div>
            
            <button id="analyzeBtn" class="btn" disabled style="width: 100%;">
                🔍 Analizar Emoción
            </button>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>Analizando con IA...</p>
            </div>
            
            <div id="error" class="error" style="display: none;"></div>
            <div id="result" style="display: none;"></div>
        </div>
    </div>

    <script>
        let selectedFile = null;
        
        document.getElementById('audioFile').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                selectedFile = file;
                document.getElementById('analyzeBtn').disabled = false;
                hideError();
            }
        });
        
        document.getElementById('analyzeBtn').addEventListener('click', async function() {
            if (!selectedFile) return;
            
            showLoading();
            
            try {
                const formData = new FormData();
                formData.append('audio', selectedFile);
                
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    hideLoading();
                    showResult(data);
                } else {
                    throw new Error(data.error || 'Error en el análisis');
                }
                
            } catch (error) {
                hideLoading();
                showError('Error: ' + error.message);
            }
        });
        
        function showResult(data) {
            const messages = {
                'Alegría': '🌟 ¡Qué maravilloso! Tu energía positiva es contagiosa.',
                'Tristeza': '💙 Es normal sentir tristeza. No estás solo.',
                'Ansiedad': '🌸 Respira profundo. Tienes la fuerza para superar esto.',
                'Neutral': '⚖️ Un estado equilibrado es valioso.',
                'Enojo': '🔥 El enojo es válido. Busca formas constructivas de expresarlo.'
            };
            
            const emotionEmojis = {
                'Alegría': '😊', 'Tristeza': '😢', 'Ansiedad': '😰', 'Neutral': '😐', 'Enojo': '😠'
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
        
        function showError(message) {
            document.getElementById('error').textContent = message;
            document.getElementById('error').style.display = 'block';
        }
        
        function hideError() {
            document.getElementById('error').style.display = 'none';
        }
        
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('analyzeBtn').disabled = true;
        }
        
        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('analyzeBtn').disabled = false;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_SIMPLE)

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No se encontró archivo de audio'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó archivo'}), 400
        
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Procesar audio
            features = extract_audio_features_simple(temp_path)
            if features is None:
                return jsonify({'error': 'No se pudo procesar el audio'}), 500
            
            # Analizar emoción
            result = analyze_emotion_simple(features)
            if result is None:
                return jsonify({'error': 'Error en el análisis emocional'}), 500
            
            result['timestamp'] = datetime.now().isoformat()
            return jsonify(result)
        
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': 'simple'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
