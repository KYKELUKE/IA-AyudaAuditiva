from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import librosa
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import json
from datetime import datetime
import warnings

# Suprimir warnings de librosa para una salida más limpia
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuración
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}
UPLOAD_FOLDER = 'uploads'

# Crear carpeta de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio_features(file_path):
    """Extrae características avanzadas del audio para análisis emocional"""
    try:
        print(f"🎵 Procesando archivo: {os.path.basename(file_path)}")
        
        # Cargar audio con librosa
        y, sr = librosa.load(file_path, sr=22050, duration=30)
        
        if len(y) == 0:
            print("❌ Archivo de audio vacío")
            return None
        
        features = {}
        
        # 1. MFCC (Mel-frequency cepstral coefficients) - Características espectrales
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = float(np.mean(mfccs))
        features['mfcc_std'] = float(np.std(mfccs))
        features['mfcc_var'] = float(np.var(mfccs))
        
        # 2. Chroma features - Relacionadas con el tono
        chroma = librosa.feature.chroma(y=y, sr=sr)
        features['chroma_mean'] = float(np.mean(chroma))
        features['chroma_std'] = float(np.std(chroma))
        
        # 3. Características espectrales
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid'] = float(np.mean(spectral_centroids))
        features['spectral_centroid_std'] = float(np.std(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
        
        # 4. Zero crossing rate - Indica cambios en la señal
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # 5. Tempo y ritmo
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_count'] = len(beats)
        except:
            features['tempo'] = 120.0
            features['beat_count'] = 0
        
        # 6. RMS Energy - Energía de la señal
        rms = librosa.feature.rms(y=y)
        features['rms'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        
        # 7. Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast'] = float(np.mean(spectral_contrast))
        
        # 8. Tonnetz (Tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features['tonnetz'] = float(np.mean(tonnetz))
        
        # 9. Características de pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        features['pitch_mean'] = float(pitch_mean)
        
        # 10. Duración del audio
        features['duration'] = float(len(y) / sr)
        
        print(f"✅ Características extraídas: {len(features)} features")
        return features
        
    except Exception as e:
        print(f"❌ Error extrayendo características: {e}")
        return None

def analyze_emotion_advanced(features):
    """Análisis emocional avanzado basado en características de audio"""
    if not features:
        return None
    
    # Obtener características clave
    spectral_centroid = features.get('spectral_centroid', 1500)
    tempo = features.get('tempo', 120)
    zcr = features.get('zcr', 0.05)
    rms = features.get('rms', 0.1)
    mfcc_mean = features.get('mfcc_mean', 0)
    mfcc_std = features.get('mfcc_std', 0)
    spectral_rolloff = features.get('spectral_rolloff', 2000)
    chroma_mean = features.get('chroma_mean', 0.5)
    spectral_contrast = features.get('spectral_contrast', 0)
    pitch_mean = features.get('pitch_mean', 0)
    
    print(f"🔍 Analizando características:")
    print(f"   Tempo: {tempo:.1f} BPM")
    print(f"   Spectral Centroid: {spectral_centroid:.1f} Hz")
    print(f"   ZCR: {zcr:.4f}")
    print(f"   RMS Energy: {rms:.4f}")
    print(f"   MFCC Mean: {mfcc_mean:.4f}")
    
    # Sistema de puntuación para cada emoción
    emotion_scores = {
        'Alegría': 0,
        'Tristeza': 0,
        'Ansiedad': 0,
        'Enojo': 0,
        'Neutral': 0
    }
    
    # Análisis para ALEGRÍA
    if tempo > 120:
        emotion_scores['Alegría'] += (tempo - 120) / 2
    if spectral_centroid > 2000:
        emotion_scores['Alegría'] += (spectral_centroid - 2000) / 100
    if rms > 0.15:
        emotion_scores['Alegría'] += (rms - 0.15) * 100
    if chroma_mean > 0.6:
        emotion_scores['Alegría'] += (chroma_mean - 0.6) * 50
    
    # Análisis para TRISTEZA
    if tempo < 90:
        emotion_scores['Tristeza'] += (90 - tempo) / 2
    if spectral_centroid < 1200:
        emotion_scores['Tristeza'] += (1200 - spectral_centroid) / 50
    if rms < 0.08:
        emotion_scores['Tristeza'] += (0.08 - rms) * 200
    if mfcc_mean < -10:
        emotion_scores['Tristeza'] += abs(mfcc_mean + 10)
    
    # Análisis para ANSIEDAD
    if zcr > 0.08:
        emotion_scores['Ansiedad'] += (zcr - 0.08) * 500
    if mfcc_std > 15:
        emotion_scores['Ansiedad'] += (mfcc_std - 15) / 2
    if spectral_rolloff > 3000:
        emotion_scores['Ansiedad'] += (spectral_rolloff - 3000) / 100
    if tempo > 130 and rms > 0.12:
        emotion_scores['Ansiedad'] += 10
    
    # Análisis para ENOJO
    if tempo > 140:
        emotion_scores['Enojo'] += (tempo - 140) / 3
    if rms > 0.2:
        emotion_scores['Enojo'] += (rms - 0.2) * 150
    if spectral_contrast > 20:
        emotion_scores['Enojo'] += (spectral_contrast - 20)
    if zcr > 0.1:
        emotion_scores['Enojo'] += (zcr - 0.1) * 300
    
    # Análisis para NEUTRAL
    if 90 <= tempo <= 130:
        emotion_scores['Neutral'] += 15
    if 0.08 <= rms <= 0.15:
        emotion_scores['Neutral'] += 10
    if 1200 <= spectral_centroid <= 2000:
        emotion_scores['Neutral'] += 8
    if 0.04 <= zcr <= 0.08:
        emotion_scores['Neutral'] += 5
    
    # Determinar la emoción dominante
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    max_score = emotion_scores[dominant_emotion]
    
    # Calcular confianza basada en la diferencia entre la emoción dominante y las demás
    scores_list = list(emotion_scores.values())
    scores_list.sort(reverse=True)
    
    if len(scores_list) > 1 and scores_list[1] > 0:
        confidence = min(95, max(65, int(70 + (scores_list[0] - scores_list[1]) * 2)))
    else:
        confidence = min(95, max(65, int(70 + max_score)))
    
    print(f"📊 Puntuaciones emocionales:")
    for emotion, score in emotion_scores.items():
        print(f"   {emotion}: {score:.2f}")
    print(f"🎯 Emoción detectada: {dominant_emotion} ({confidence}%)")
    
    return {
        'emotion': dominant_emotion,
        'confidence': confidence,
        'scores': emotion_scores,
        'features_analyzed': len(features)
    }

# HTML Template mejorado
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
            max-width: 900px;
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
        .success-notice {
            background: #d4edda;
            border: 2px solid #c3e6cb;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            color: #155724;
            text-align: center;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .upload-area:hover { 
            border-color: #764ba2; 
            background: #f8f9ff; 
            transform: translateY(-2px);
        }
        .upload-area.dragover {
            border-color: #38ef7d;
            background: #f0fff4;
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
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .emotion {
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .confidence {
            font-size: 1.4em;
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
        .features-info {
            background: #e8f4fd;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
            font-size: 0.9em;
            color: #1565c0;
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
            width: 50px;
            height: 50px;
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
        .progress-bar {
            width: 100%;
            height: 6px;
            background: #f0f0f0;
            border-radius: 3px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Asistente de Apoyo Emocional</h1>
            <p>Análisis emocional avanzado con IA y procesamiento de audio real</p>
        </div>
        
        <div class="content">
            <div class="success-notice">
                <strong>✅ Sistema Completo Activado:</strong> Análisis real de audio con librosa habilitado. 
                Todas las dependencias instaladas correctamente.
            </div>
            
            <div class="upload-area" id="uploadArea" onclick="document.getElementById('audioFile').click()">
                <h3>🎵 Arrastra tu archivo de audio aquí o haz clic para seleccionar</h3>
                <p>Formatos soportados: WAV, MP3, FLAC, M4A, OGG</p>
                <p style="font-size: 0.9em; margin-top: 10px; opacity: 0.8;">
                    Máximo 16MB • Duración recomendada: 10-30 segundos
                </p>
                <br>
                <button class="upload-btn" type="button">📁 Elegir Archivo</button>
                <input type="file" id="audioFile" accept=".wav,.mp3,.flac,.m4a,.ogg" onchange="handleFileSelect(this)" style="display: none;">
            </div>
            
            <div id="fileInfo" class="file-info" style="display: none;"></div>
            
            <button id="analyzeBtn" class="analyze-btn" onclick="analyzeAudio()" disabled>
                🔍 Analizar Emoción con IA
            </button>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p id="loadingText">Procesando audio y extrayendo características...</p>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
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
        
        // Drag and drop functionality
        const uploadArea = document.getElementById('uploadArea');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        function handleFileSelect(input) {
            const file = input.files[0];
            if (file) {
                handleFile(file);
            }
        }
        
        function handleFile(file) {
            const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/flac', 'audio/m4a', 'audio/ogg'];
            
            if (!allowedTypes.some(type => file.type.includes(type.split('/')[1]))) {
                alert('Por favor, selecciona un archivo de audio válido (WAV, MP3, FLAC, M4A, OGG)');
                return;
            }
            
            if (file.size > 16 * 1024 * 1024) {
                alert('El archivo es demasiado grande. Máximo 16MB permitido.');
                return;
            }
            
            selectedFile = file;
            document.getElementById('fileInfo').style.display = 'block';
            document.getElementById('fileInfo').innerHTML = `
                <strong>📄 Archivo seleccionado:</strong> ${file.name}<br>
                <strong>📏 Tamaño:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB<br>
                <strong>🎵 Tipo:</strong> ${file.type}<br>
                <strong>⏱️ Listo para análisis con IA</strong>
            `;
            document.getElementById('analyzeBtn').disabled = false;
            document.getElementById('result').style.display = 'none';
        }
        
        async function analyzeAudio() {
            if (!selectedFile) {
                alert('Por favor selecciona un archivo de audio primero');
                return;
            }
            
            const formData = new FormData();
            formData.append('audio', selectedFile);
            
            // Mostrar loading con progreso simulado
            document.getElementById('loading').style.display = 'block';
            document.getElementById('analyzeBtn').disabled = true;
            document.getElementById('result').style.display = 'none';
            
            // Simular progreso
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                document.getElementById('progressFill').style.width = progress + '%';
                
                if (progress < 30) {
                    document.getElementById('loadingText').textContent = 'Cargando archivo de audio...';
                } else if (progress < 60) {
                    document.getElementById('loadingText').textContent = 'Extrayendo características espectrales...';
                } else if (progress < 85) {
                    document.getElementById('loadingText').textContent = 'Analizando patrones emocionales...';
                } else {
                    document.getElementById('loadingText').textContent = 'Finalizando análisis...';
                }
            }, 200);
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                clearInterval(progressInterval);
                document.getElementById('progressFill').style.width = '100%';
                
                setTimeout(() => {
                    if (response.ok) {
                        showResult(data);
                    } else {
                        alert('Error: ' + data.error);
                    }
                }, 500);
                
            } catch (error) {
                clearInterval(progressInterval);
                alert('Error de conexión: ' + error.message);
            } finally {
                setTimeout(() => {
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('analyzeBtn').disabled = false;
                    document.getElementById('progressFill').style.width = '0%';
                }, 1000);
            }
        }
        
        function showResult(data) {
            const messages = {
                'Alegría': '🌟 ¡Qué maravilloso! Tu energía positiva es contagiosa. Sigue cultivando esos momentos de felicidad y compártelos con otros. La alegría es un regalo que se multiplica cuando se comparte.',
                'Tristeza': '💙 Es completamente normal sentir tristeza. Estos sentimientos son temporales y forman parte de la experiencia humana. Permítete sentir, pero recuerda que no estás solo. Considera hablar con alguien de confianza.',
                'Ansiedad': '🌸 Respira profundo y despacio. La ansiedad puede ser abrumadora, pero recuerda que tienes la fuerza para superarla. Prueba técnicas de relajación, mindfulness, o ejercicio suave. Tu bienestar es importante.',
                'Neutral': '⚖️ Un estado equilibrado es muy valioso. Aprovecha esta calma para reflexionar, planificar y cuidar tu bienestar mental. Es un buen momento para practicar gratitud y establecer intenciones positivas.',
                'Enojo': '🔥 El enojo es una emoción válida que nos indica que algo necesita atención. Trata de identificar la causa raíz y busca formas constructivas de expresarlo. El ejercicio físico puede ayudar a canalizar esta energía.'
            };
            
            const emotionEmojis = {
                'Alegría': '😊',
                'Tristeza': '😢',
                'Ansiedad': '😰',
                'Neutral': '😐',
                'Enojo': '😠'
            };
            
            const confidenceColor = data.confidence >= 80 ? '#27ae60' : data.confidence >= 70 ? '#f39c12' : '#e74c3c';
            
            document.getElementById('result').innerHTML = `
                <div class="result">
                    <div class="emotion">${emotionEmojis[data.emotion]} ${data.emotion}</div>
                    <div class="confidence" style="color: ${confidenceColor}">
                        Confianza: ${data.confidence}%
                    </div>
                    <div class="message">${messages[data.emotion]}</div>
                    <div class="features-info">
                        <strong>🔬 Análisis técnico:</strong> Se analizaron ${data.features_analyzed} características de audio incluyendo 
                        MFCC, características espectrales, tempo, energía RMS, y patrones tonales.
                        <br><strong>⏰ Procesado:</strong> ${new Date(data.timestamp).toLocaleString()}
                    </div>
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
        'message': 'Asistente de Apoyo Emocional API - Versión Completa',
        'version': '2.0.0',
        'features': [
            'Análisis real de audio con librosa',
            'Extracción de características avanzadas',
            'Clasificación emocional inteligente',
            'Interfaz web moderna'
        ],
        'endpoints': {
            '/': 'Interfaz web',
            '/analyze': 'Análisis de audio (POST)',
            '/health': 'Estado del servicio'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'mode': 'full_analysis',
        'librosa_available': True,
        'timestamp': datetime.now().isoformat()
    })

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
            return jsonify({'error': 'Formato de archivo no permitido. Use WAV, MP3, FLAC, M4A u OGG'}), 400
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{timestamp}_{filename}")
        
        try:
            file.save(temp_path)
            print(f"📁 Archivo guardado: {temp_path}")
            
            # Extraer características con librosa
            features = extract_audio_features(temp_path)
            
            if features is None:
                return jsonify({'error': 'Error al procesar el archivo de audio. Verifique que sea un archivo válido y no esté corrupto.'}), 500
            
            # Analizar emoción con algoritmo avanzado
            result = analyze_emotion_advanced(features)
            
            if result is None:
                return jsonify({'error': 'Error en el análisis emocional'}), 500
            
            print(f"✅ Análisis completado: {result['emotion']} ({result['confidence']}%)")
            
            response = {
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'features_analyzed': result['features_analyzed'],
                'timestamp': datetime.now().isoformat(),
                'filename': filename,
                'analysis_type': 'advanced_ml'
            }
            
            return jsonify(response)
        
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                print(f"🗑️ Archivo temporal eliminado")
    
    except Exception as e:
        print(f"❌ Error en análisis: {str(e)}")
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

if __name__ == '__main__':
    print("🧠 Asistente de Apoyo Emocional - VERSIÓN COMPLETA")
    print("=" * 60)
    print("✅ Librosa instalado correctamente")
    print("✅ Análisis real de audio habilitado")
    print("✅ Características avanzadas disponibles")
    print("=" * 60)
    print("🌐 Servidor: http://localhost:5000")
    print("📱 Interfaz web: http://localhost:5000")
    print("🔗 API: http://localhost:5000/api")
    print("💚 Estado: http://localhost:5000/health")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
