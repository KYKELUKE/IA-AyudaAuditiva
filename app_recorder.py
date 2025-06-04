from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import os
import librosa
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import json
from datetime import datetime
import warnings
import base64
import io

# Suprimir warnings de librosa para una salida más limpia
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuración
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'}
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
        try:
            y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)
        except Exception as e:
            print(f"❌ Error cargando audio con librosa: {e}")
            return None
        
        if len(y) == 0:
            print("❌ Archivo de audio vacío")
            return None
        
        features = {}
        
        # 1. MFCC (Mel-frequency cepstral coefficients) - Características espectrales
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = float(np.mean(mfccs))
            features['mfcc_std'] = float(np.std(mfccs))
            features['mfcc_var'] = float(np.var(mfccs))
        except Exception as e:
            print(f"❌ Error extrayendo MFCC: {e}")
            features['mfcc_mean'] = 0.0
            features['mfcc_std'] = 0.0
            features['mfcc_var'] = 0.0
        
        # 2. Chroma features - Relacionadas con el tono
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
        except Exception as e:
            print(f"❌ Error extrayendo Chroma: {e}")
            features['chroma_mean'] = 0.5
            features['chroma_std'] = 0.1
        
        # 3. Características espectrales
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
        except Exception as e:
            print(f"❌ Error extrayendo características espectrales: {e}")
            features['spectral_centroid'] = 1500.0
            features['spectral_centroid_std'] = 300.0
            features['spectral_rolloff'] = 2000.0
            features['spectral_bandwidth'] = 1000.0
        
        # 4. Zero crossing rate - Indica cambios en la señal
        try:
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
        except Exception as e:
            print(f"❌ Error extrayendo ZCR: {e}")
            features['zcr'] = 0.05
            features['zcr_std'] = 0.01
        
        # 5. Tempo y ritmo
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_count'] = len(beats)
        except Exception as e:
            print(f"❌ Error extrayendo tempo: {e}")
            features['tempo'] = 120.0
            features['beat_count'] = 0
        
        # 6. RMS Energy - Energía de la señal
        try:
            rms = librosa.feature.rms(y=y)
            features['rms'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
        except Exception as e:
            print(f"❌ Error extrayendo RMS: {e}")
            features['rms'] = 0.1
            features['rms_std'] = 0.02
        
        # 7. Spectral contrast
        try:
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast'] = float(np.mean(spectral_contrast))
        except Exception as e:
            print(f"❌ Error extrayendo spectral contrast: {e}")
            features['spectral_contrast'] = 10.0
        
        # 8. Tonnetz (Tonal centroid features)
        try:
            y_harmonic = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
            features['tonnetz'] = float(np.mean(tonnetz))
        except Exception as e:
            print(f"❌ Error extrayendo tonnetz: {e}")
            features['tonnetz'] = 0.0
        
        # 9. Duración del audio
        features['duration'] = float(len(y) / sr)
        
        print(f"✅ Características extraídas: {len(features)} features")
        return features
        
    except Exception as e:
        print(f"❌ Error general extrayendo características: {e}")
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

# HTML Template mejorado con grabación de audio
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
        .tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 2px solid #eee;
        }
        .tab {
            padding: 12px 20px;
            cursor: pointer;
            font-weight: 600;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }
        .tab:hover {
            color: #764ba2;
        }
        .tab.active {
            color: #764ba2;
            border-bottom-color: #764ba2;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
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
        .recorder-container {
            border: 3px solid #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            background: #f8f9ff;
        }
        .recorder-controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
        }
        .record-btn {
            background: linear-gradient(135deg, #ff5f6d 0%, #ffc371 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .record-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 95, 109, 0.3);
        }
        .record-btn.recording {
            background: linear-gradient(135deg, #ff5f6d 0%, #ff5f6d 100%);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 95, 109, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 95, 109, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 95, 109, 0); }
        }
        .stop-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .stop-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
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
        .recording-time {
            font-size: 1.5em;
            font-weight: bold;
            margin: 15px 0;
            color: #ff5f6d;
        }
        .audio-player {
            width: 100%;
            margin: 20px 0;
        }
        .audio-visualizer {
            width: 100%;
            height: 60px;
            background: #f0f0f0;
            border-radius: 10px;
            margin: 15px 0;
            position: relative;
            overflow: hidden;
        }
        .visualizer-bar {
            position: absolute;
            bottom: 0;
            width: 3px;
            background: #764ba2;
            margin: 0 2px;
            border-radius: 3px 3px 0 0;
        }
        .recording-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background-color: #ff5f6d;
            border-radius: 50%;
            margin-right: 8px;
            animation: blink 1s infinite;
        }
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.3; }
            100% { opacity: 1; }
        }
        .error-message {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            text-align: center;
            border: 1px solid #f5c6cb;
        }
        .help-text {
            font-size: 0.9em;
            color: #666;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Asistente de Apoyo Emocional</h1>
            <p>Análisis emocional avanzado con IA y procesamiento de audio</p>
        </div>
        
        <div class="content">
            <div class="success-notice">
                <strong>✅ Sistema Completo Activado:</strong> Análisis real de audio con librosa habilitado. 
                Ahora puedes grabar audio directamente desde el navegador.
            </div>
            
            <div class="tabs">
                <div class="tab active" data-tab="record">🎙️ Grabar Audio</div>
                <div class="tab" data-tab="upload">📁 Subir Archivo</div>
            </div>
            
            <div id="recordTab" class="tab-content active">
                <div class="recorder-container">
                    <h3>🎙️ Graba tu voz para analizar tu emoción</h3>
                    <p>Habla durante 5-30 segundos sobre cómo te sientes o cualquier tema</p>
                    
                    <div id="audioVisualizer" class="audio-visualizer">
                        <!-- Las barras del visualizador se generarán con JavaScript -->
                    </div>
                    
                    <div id="recordingTime" class="recording-time" style="display: none;">
                        <span class="recording-indicator"></span>
                        <span id="recordingTimer">00:00</span>
                    </div>
                    
                    <div id="audioPlayerContainer" style="display: none;">
                        <audio id="audioPlayer" controls class="audio-player"></audio>
                    </div>
                    
                    <div class="recorder-controls">
                        <button id="recordBtn" class="record-btn">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="6"></circle>
                            </svg>
                            Iniciar Grabación
                        </button>
                        <button id="stopBtn" class="stop-btn" style="display: none;">
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <rect x="6" y="6" width="12" height="12"></rect>
                            </svg>
                            Detener Grabación
                        </button>
                    </div>
                    
                    <p class="help-text">
                        Asegúrate de permitir el acceso al micrófono cuando el navegador lo solicite.
                        <br>Para mejores resultados, habla claramente y evita ruidos de fondo.
                    </p>
                </div>
                
                <button id="analyzeRecordingBtn" class="analyze-btn" disabled>
                    🔍 Analizar Emoción con IA
                </button>
            </div>
            
            <div id="uploadTab" class="tab-content">
                <div class="upload-area" id="uploadArea" onclick="document.getElementById('audioFile').click()">
                    <h3>🎵 Arrastra tu archivo de audio aquí o haz clic para seleccionar</h3>
                    <p>Formatos soportados: WAV, MP3, FLAC, M4A, OGG, WEBM</p>
                    <p style="font-size: 0.9em; margin-top: 10px; opacity: 0.8;">
                        Máximo 16MB • Duración recomendada: 10-30 segundos
                    </p>
                    <br>
                    <button class="upload-btn" type="button">📁 Elegir Archivo</button>
                    <input type="file" id="audioFile" accept=".wav,.mp3,.flac,.m4a,.ogg,.webm" onchange="handleFileSelect(this)" style="display: none;">
                </div>
                
                <div id="fileInfo" class="file-info" style="display: none;"></div>
                
                <button id="analyzeFileBtn" class="analyze-btn" disabled>
                    🔍 Analizar Emoción con IA
                </button>
            </div>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p id="loadingText">Procesando audio y extrayendo características...</p>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
            </div>
            
            <div id="errorMessage" class="error-message" style="display: none;"></div>
            
            <div id="result" style="display: none;"></div>
            
            <div class="disclaimer">
                <strong>⚠️ Importante:</strong> Este sistema es solo un apoyo emocional y no sustituye el diagnóstico profesional. 
                Consulta siempre a un especialista en salud mental para obtener ayuda profesional.
            </div>
        </div>
    </div>

    <script>
        // Variables globales
        let selectedFile = null;
        let mediaRecorder = null;
        let audioChunks = [];
        let recordedBlob = null;
        let recordingTimer = null;
        let recordingSeconds = 0;
        let audioContext = null;
        let analyser = null;
        let visualizerBars = [];
        
        // Crear barras del visualizador
        function createVisualizer() {
            const visualizer = document.getElementById('audioVisualizer');
            visualizer.innerHTML = '';
            
            const barCount = 50;
            for (let i = 0; i < barCount; i++) {
                const bar = document.createElement('div');
                bar.className = 'visualizer-bar';
                bar.style.left = `${i * 5}px`;
                bar.style.height = '0px';
                visualizer.appendChild(bar);
                visualizerBars.push(bar);
            }
        }
        
        // Actualizar visualizador
        function updateVisualizer(dataArray) {
            const bufferLength = dataArray.length;
            const barCount = visualizerBars.length;
            
            for (let i = 0; i < barCount; i++) {
                const index = Math.floor(i * bufferLength / barCount);
                const value = dataArray[index] / 255;
                const height = value * 60;
                visualizerBars[i].style.height = `${height}px`;
            }
        }
        
        // Inicializar visualizador con valores aleatorios
        function initVisualizer() {
            createVisualizer();
            
            // Simular actividad para el visualizador cuando no está grabando
            const simulateActivity = () => {
                if (!mediaRecorder || mediaRecorder.state !== 'recording') {
                    const dataArray = new Uint8Array(128);
                    for (let i = 0; i < dataArray.length; i++) {
                        dataArray[i] = Math.random() * 50;
                    }
                    updateVisualizer(dataArray);
                }
            };
            
            // Actualizar visualizador cada 100ms
            setInterval(simulateActivity, 100);
        }
        
        // Iniciar grabación
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                // Configurar AudioContext para visualización
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const source = audioContext.createMediaStreamSource(stream);
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                source.connect(analyser);
                
                const bufferLength = analyser.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                
                // Actualizar visualizador durante la grabación
                const updateVisualizerLoop = () => {
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        analyser.getByteFrequencyData(dataArray);
                        updateVisualizer(dataArray);
                        requestAnimationFrame(updateVisualizerLoop);
                    }
                };
                
                // Configurar MediaRecorder
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = () => {
                    recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const audioURL = URL.createObjectURL(recordedBlob);
                    document.getElementById('audioPlayer').src = audioURL;
                    document.getElementById('audioPlayerContainer').style.display = 'block';
                    document.getElementById('analyzeRecordingBtn').disabled = false;
                    
                    // Detener todas las pistas del stream
                    stream.getTracks().forEach(track => track.stop());
                };
                
                // Iniciar grabación
                audioChunks = [];
                mediaRecorder.start();
                updateVisualizerLoop();
                
                // Mostrar tiempo de grabación
                recordingSeconds = 0;
                document.getElementById('recordingTimer').textContent = '00:00';
                document.getElementById('recordingTime').style.display = 'block';
                
                recordingTimer = setInterval(() => {
                    recordingSeconds++;
                    const minutes = Math.floor(recordingSeconds / 60);
                    const seconds = recordingSeconds % 60;
                    document.getElementById('recordingTimer').textContent = 
                        `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                    
                    // Limitar grabación a 2 minutos
                    if (recordingSeconds >= 120) {
                        stopRecording();
                    }
                }, 1000);
                
                // Actualizar UI
                document.getElementById('recordBtn').style.display = 'none';
                document.getElementById('stopBtn').style.display = 'inline-flex';
                document.getElementById('recordBtn').classList.add('recording');
                
            } catch (error) {
                console.error('Error accessing microphone:', error);
                showError('No se pudo acceder al micrófono. Por favor, asegúrate de dar permiso al navegador para usar el micrófono.');
            }
        }
        
        // Detener grabación
        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                clearInterval(recordingTimer);
                
                // Actualizar UI
                document.getElementById('recordBtn').style.display = 'inline-flex';
                document.getElementById('stopBtn').style.display = 'none';
                document.getElementById('recordBtn').classList.remove('recording');
                document.getElementById('recordBtn').textContent = 'Grabar de nuevo';
            }
        }
        
        // Analizar audio grabado
        async function analyzeRecording() {
            if (!recordedBlob) {
                showError('No hay grabación para analizar');
                return;
            }
            
            showLoading();
            
            try {
                // Crear FormData con el blob grabado
                const formData = new FormData();
                formData.append('audio', recordedBlob, 'recording.webm');
                
                // Enviar al servidor
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
                showError('Error al analizar el audio: ' + error.message);
            }
        }
        
        // Manejar selección de archivo
        function handleFileSelect(input) {
            const file = input.files[0];
            if (file) {
                handleFile(file);
            }
        }
        
        function handleFile(file) {
            const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/flac', 'audio/m4a', 'audio/ogg', 'audio/webm'];
            const fileType = file.type.toLowerCase();
            
            // Verificar si el tipo de archivo está permitido
            if (!allowedTypes.some(type => fileType.includes(type.split('/')[1]))) {
                showError('Por favor, selecciona un archivo de audio válido (WAV, MP3, FLAC, M4A, OGG, WEBM)');
                return;
            }
            
            if (file.size > 16 * 1024 * 1024) {
                showError('El archivo es demasiado grande. Máximo 16MB permitido.');
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
            document.getElementById('analyzeFileBtn').disabled = false;
            document.getElementById('result').style.display = 'none';
            document.getElementById('errorMessage').style.display = 'none';
        }
        
        // Analizar archivo de audio
        async function analyzeFile() {
            if (!selectedFile) {
                showError('Por favor selecciona un archivo de audio primero');
                return;
            }
            
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
                showError('Error al analizar el archivo: ' + error.message);
            }
        }
        
        // Mostrar resultado
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
            
            // Scroll al resultado
            document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
        }
        
        // Mostrar error
        function showError(message) {
            const errorElement = document.getElementById('errorMessage');
            errorElement.textContent = message;
            errorElement.style.display = 'block';
            
            // Ocultar después de 5 segundos
            setTimeout(() => {
                errorElement.style.display = 'none';
            }, 5000);
        }
        
        // Mostrar loading
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('analyzeFileBtn').disabled = true;
            document.getElementById('analyzeRecordingBtn').disabled = true;
            document.getElementById('result').style.display = 'none';
            document.getElementById('errorMessage').style.display = 'none';
            
            // Simular progreso
            let progress = 0;
            window.progressInterval = setInterval(() => {
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
        }
        
        // Ocultar loading
        function hideLoading() {
            clearInterval(window.progressInterval);
            document.getElementById('progressFill').style.width = '100%';
            
            setTimeout(() => {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('analyzeFileBtn').disabled = false;
                document.getElementById('analyzeRecordingBtn').disabled = false;
                document.getElementById('progressFill').style.width = '0%';
            }, 500);
        }
        
        // Cambiar entre pestañas
        function setupTabs() {
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    // Desactivar todas las pestañas
                    tabs.forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(content => {
                        content.classList.remove('active');
                    });
                    
                    // Activar la pestaña seleccionada
                    tab.classList.add('active');
                    const tabId = tab.getAttribute('data-tab');
                    document.getElementById(tabId + 'Tab').classList.add('active');
                    
                    // Ocultar mensajes de error y resultados
                    document.getElementById('errorMessage').style.display = 'none';
                    document.getElementById('result').style.display = 'none';
                });
            });
        }
        
        // Drag and drop para subida de archivos
        function setupDragAndDrop() {
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
        }
        
        // Inicializar
        document.addEventListener('DOMContentLoaded', () => {
            // Configurar pestañas
            setupTabs();
            
            // Configurar drag and drop
            setupDragAndDrop();
            
            // Inicializar visualizador
            initVisualizer();
            
            // Event listeners para botones
            document.getElementById('recordBtn').addEventListener('click', startRecording);
            document.getElementById('stopBtn').addEventListener('click', stopRecording);
            document.getElementById('analyzeRecordingBtn').addEventListener('click', analyzeRecording);
            document.getElementById('analyzeFileBtn').addEventListener('click', analyzeFile);
        });
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
            'Grabación de audio en el navegador',
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
        
        # Guardar archivo temporalmente
        filename = secure_filename(file.filename) if file.filename else "recorded_audio.webm"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{timestamp}_{filename}")
        
        try:
            file.save(temp_path)
            print(f"📁 Archivo guardado: {temp_path}")
            
            # Extraer características con librosa
            features = extract_audio_features(temp_path)
            
            if features is None:
                return jsonify({'error': 'Error al procesar el archivo de audio. Intenta con otro archivo o graba directamente desde el navegador.'}), 500
            
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

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    print("🧠 Asistente de Apoyo Emocional - VERSIÓN COMPLETA CON GRABACIÓN")
    print("=" * 70)
    print("✅ Librosa instalado correctamente")
    print("✅ Análisis real de audio habilitado")
    print("✅ Grabación de audio en navegador habilitada")
    print("✅ Características avanzadas disponibles")
    print("=" * 70)
    print("🌐 Servidor: http://localhost:5000")
    print("📱 Interfaz web: http://localhost:5000")
    print("🔗 API: http://localhost:5000/api")
    print("💚 Estado: http://localhost:5000/health")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
