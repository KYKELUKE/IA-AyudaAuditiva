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
import subprocess
import sys

# Suprimir warnings de librosa para una salida más limpia
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuración
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max (aumentado)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm', 'aac'}
UPLOAD_FOLDER = 'uploads'

# Crear carpeta de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_audio_to_wav(input_path, output_path):
    """Convierte cualquier formato de audio a WAV usando ffmpeg si está disponible"""
    try:
        # Intentar usar ffmpeg para conversión
        subprocess.run([
            'ffmpeg', '-i', input_path, '-ar', '22050', '-ac', '1', 
            '-f', 'wav', output_path, '-y'
        ], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg no disponible, usando librosa para conversión")
        return False

def extract_audio_features_robust(file_path):
    """Extrae características de audio con múltiples métodos de respaldo"""
    try:
        print(f"🎵 Procesando archivo: {os.path.basename(file_path)}")
        
        # Método 1: Intentar cargar directamente con librosa
        y, sr = None, None
        
        try:
            print("📥 Método 1: Carga directa con librosa...")
            y, sr = librosa.load(file_path, sr=22050, mono=True, duration=60)
            print(f"✅ Carga exitosa: {len(y)} muestras a {sr} Hz")
        except Exception as e:
            print(f"❌ Método 1 falló: {e}")
            
            # Método 2: Intentar conversión a WAV temporal
            try:
                print("📥 Método 2: Conversión a WAV temporal...")
                temp_wav = file_path.replace(os.path.splitext(file_path)[1], '_temp.wav')
                
                if convert_audio_to_wav(file_path, temp_wav):
                    y, sr = librosa.load(temp_wav, sr=22050, mono=True, duration=60)
                    os.unlink(temp_wav)  # Limpiar archivo temporal
                    print(f"✅ Conversión exitosa: {len(y)} muestras a {sr} Hz")
                else:
                    raise Exception("Conversión FFmpeg falló")
                    
            except Exception as e2:
                print(f"❌ Método 2 falló: {e2}")
                
                # Método 3: Intentar con diferentes parámetros de librosa
                try:
                    print("📥 Método 3: Librosa con parámetros alternativos...")
                    y, sr = librosa.load(file_path, sr=None, mono=True, duration=60)
                    if sr != 22050:
                        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                        sr = 22050
                    print(f"✅ Carga alternativa exitosa: {len(y)} muestras a {sr} Hz")
                except Exception as e3:
                    print(f"❌ Método 3 falló: {e3}")
                    return None
        
        if y is None or len(y) == 0:
            print("❌ No se pudo extraer audio válido")
            return None
        
        # Normalizar audio
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        features = {}
        
        # Extraer características con manejo de errores individual
        print("🔬 Extrayendo características...")
        
        # 1. MFCC (Mel-frequency cepstral coefficients)
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = float(np.mean(mfccs))
            features['mfcc_std'] = float(np.std(mfccs))
            features['mfcc_var'] = float(np.var(mfccs))
            print("✅ MFCC extraído")
        except Exception as e:
            print(f"⚠️ Error MFCC: {e}")
            features['mfcc_mean'] = 0.0
            features['mfcc_std'] = 1.0
            features['mfcc_var'] = 1.0
        
        # 2. Características espectrales básicas
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            print("✅ Spectral centroid extraído")
        except Exception as e:
            print(f"⚠️ Error spectral centroid: {e}")
            features['spectral_centroid'] = 1500.0
            features['spectral_centroid_std'] = 300.0
        
        # 3. Zero crossing rate
        try:
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))
            print("✅ ZCR extraído")
        except Exception as e:
            print(f"⚠️ Error ZCR: {e}")
            features['zcr'] = 0.05
            features['zcr_std'] = 0.01
        
        # 4. RMS Energy
        try:
            rms = librosa.feature.rms(y=y)
            features['rms'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            print("✅ RMS extraído")
        except Exception as e:
            print(f"⚠️ Error RMS: {e}")
            features['rms'] = 0.1
            features['rms_std'] = 0.02
        
        # 5. Tempo (con manejo de errores robusto)
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo) if not np.isnan(tempo) else 120.0
            features['beat_count'] = len(beats)
            print(f"✅ Tempo extraído: {features['tempo']:.1f} BPM")
        except Exception as e:
            print(f"⚠️ Error tempo: {e}")
            features['tempo'] = 120.0
            features['beat_count'] = 0
        
        # 6. Características adicionales simples
        try:
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
            
            print("✅ Características espectrales adicionales extraídas")
        except Exception as e:
            print(f"⚠️ Error características espectrales: {e}")
            features['spectral_rolloff'] = 2000.0
            features['spectral_bandwidth'] = 1000.0
        
        # 7. Chroma (simplificado)
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = float(np.mean(chroma))
            features['chroma_std'] = float(np.std(chroma))
            print("✅ Chroma extraído")
        except Exception as e:
            print(f"⚠️ Error chroma: {e}")
            features['chroma_mean'] = 0.5
            features['chroma_std'] = 0.1
        
        # 8. Duración y estadísticas básicas
        features['duration'] = float(len(y) / sr)
        features['sample_rate'] = float(sr)
        features['samples'] = len(y)
        
        # 9. Estadísticas simples del audio
        features['audio_mean'] = float(np.mean(y))
        features['audio_std'] = float(np.std(y))
        features['audio_max'] = float(np.max(y))
        features['audio_min'] = float(np.min(y))
        
        print(f"✅ Extracción completada: {len(features)} características")
        return features
        
    except Exception as e:
        print(f"❌ Error general en extracción: {e}")
        return None

def analyze_emotion_simple_robust(features):
    """Análisis emocional simplificado y robusto"""
    if not features:
        return None
    
    print("🧠 Iniciando análisis emocional...")
    
    # Obtener características con valores por defecto
    spectral_centroid = features.get('spectral_centroid', 1500)
    tempo = features.get('tempo', 120)
    zcr = features.get('zcr', 0.05)
    rms = features.get('rms', 0.1)
    mfcc_mean = features.get('mfcc_mean', 0)
    chroma_mean = features.get('chroma_mean', 0.5)
    duration = features.get('duration', 10)
    audio_std = features.get('audio_std', 0.1)
    
    print(f"📊 Características clave:")
    print(f"   Tempo: {tempo:.1f} BPM")
    print(f"   Spectral Centroid: {spectral_centroid:.1f} Hz")
    print(f"   ZCR: {zcr:.4f}")
    print(f"   RMS Energy: {rms:.4f}")
    print(f"   Duración: {duration:.1f}s")
    
    # Sistema de puntuación simplificado
    emotion_scores = {
        'Alegría': 0,
        'Tristeza': 0,
        'Ansiedad': 0,
        'Enojo': 0,
        'Neutral': 10  # Valor base para neutral
    }
    
    # Análisis basado en tempo
    if tempo > 130:
        emotion_scores['Alegría'] += 15
        emotion_scores['Ansiedad'] += 10
    elif tempo > 110:
        emotion_scores['Alegría'] += 8
        emotion_scores['Neutral'] += 5
    elif tempo < 80:
        emotion_scores['Tristeza'] += 15
    elif tempo < 100:
        emotion_scores['Tristeza'] += 8
        emotion_scores['Neutral'] += 3
    
    # Análisis basado en frecuencia espectral
    if spectral_centroid > 2200:
        emotion_scores['Alegría'] += 12
        emotion_scores['Ansiedad'] += 8
    elif spectral_centroid > 1800:
        emotion_scores['Alegría'] += 6
        emotion_scores['Neutral'] += 4
    elif spectral_centroid < 1000:
        emotion_scores['Tristeza'] += 12
    elif spectral_centroid < 1400:
        emotion_scores['Tristeza'] += 6
        emotion_scores['Neutral'] += 3
    
    # Análisis basado en energía
    if rms > 0.2:
        emotion_scores['Enojo'] += 15
        emotion_scores['Alegría'] += 8
    elif rms > 0.15:
        emotion_scores['Alegría'] += 10
        emotion_scores['Enojo'] += 5
    elif rms < 0.05:
        emotion_scores['Tristeza'] += 12
    elif rms < 0.1:
        emotion_scores['Tristeza'] += 6
        emotion_scores['Neutral'] += 4
    
    # Análisis basado en variabilidad
    if zcr > 0.1:
        emotion_scores['Ansiedad'] += 15
        emotion_scores['Enojo'] += 8
    elif zcr > 0.07:
        emotion_scores['Ansiedad'] += 8
        emotion_scores['Neutral'] += 3
    elif zcr < 0.03:
        emotion_scores['Tristeza'] += 8
        emotion_scores['Neutral'] += 5
    
    # Análisis basado en variabilidad del audio
    if audio_std > 0.15:
        emotion_scores['Enojo'] += 10
        emotion_scores['Ansiedad'] += 8
    elif audio_std < 0.05:
        emotion_scores['Tristeza'] += 8
        emotion_scores['Neutral'] += 6
    
    # Determinar emoción dominante
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    max_score = emotion_scores[dominant_emotion]
    
    # Calcular confianza
    scores_list = sorted(emotion_scores.values(), reverse=True)
    if len(scores_list) > 1 and scores_list[1] > 0:
        confidence = min(95, max(65, int(65 + (scores_list[0] - scores_list[1]) * 2)))
    else:
        confidence = min(95, max(65, int(65 + max_score)))
    
    print(f"📈 Puntuaciones:")
    for emotion, score in emotion_scores.items():
        print(f"   {emotion}: {score:.1f}")
    print(f"🎯 Resultado: {dominant_emotion} ({confidence}%)")
    
    return {
        'emotion': dominant_emotion,
        'confidence': confidence,
        'scores': emotion_scores,
        'features_analyzed': len(features),
        'audio_duration': duration
    }

# HTML Template con mejor manejo de errores
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
        .tab:hover { color: #764ba2; }
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
        .recorder-container {
            border: 3px solid #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            background: #f8f9ff;
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
            margin: 10px;
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
        .troubleshooting {
            background: #e3f2fd;
            border: 1px solid #90caf9;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            color: #1565c0;
        }
        .troubleshooting h4 {
            margin-bottom: 10px;
            color: #0d47a1;
        }
        .troubleshooting ul {
            margin-left: 20px;
        }
        .troubleshooting li {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Asistente de Apoyo Emocional</h1>
            <p>Análisis emocional robusto con múltiples métodos de procesamiento</p>
        </div>
        
        <div class="content">
            <div class="success-notice">
                <strong>✅ Sistema Robusto Activado:</strong> Múltiples métodos de procesamiento de audio para máxima compatibilidad.
            </div>
            
            <div class="tabs">
                <div class="tab active" data-tab="record">🎙️ Grabar Audio</div>
                <div class="tab" data-tab="upload">📁 Subir Archivo</div>
            </div>
            
            <div id="recordTab" class="tab-content active">
                <div class="recorder-container">
                    <h3>🎙️ Graba tu voz para analizar tu emoción</h3>
                    <p>Habla durante 10-30 segundos sobre cómo te sientes</p>
                    
                    <div id="recordingTime" class="recording-time" style="display: none;">
                        🔴 Grabando: <span id="recordingTimer">00:00</span>
                    </div>
                    
                    <div id="audioPlayerContainer" style="display: none;">
                        <audio id="audioPlayer" controls class="audio-player"></audio>
                    </div>
                    
                    <div>
                        <button id="recordBtn" class="record-btn">🎙️ Iniciar Grabación</button>
                        <button id="stopBtn" class="record-btn" style="display: none;">⏹️ Detener</button>
                    </div>
                    
                    <p class="help-text">
                        💡 <strong>Consejos:</strong> Habla claramente, evita ruidos de fondo, y permite el acceso al micrófono.
                    </p>
                </div>
                
                <button id="analyzeRecordingBtn" class="analyze-btn" disabled>
                    🔍 Analizar Emoción Grabada
                </button>
            </div>
            
            <div id="uploadTab" class="tab-content">
                <div class="upload-area" onclick="document.getElementById('audioFile').click()">
                    <h3>📁 Sube tu archivo de audio</h3>
                    <p>Formatos: WAV, MP3, FLAC, M4A, OGG, WEBM, AAC</p>
                    <p class="help-text">Máximo 32MB • Duración recomendada: 10-60 segundos</p>
                    <br>
                    <button class="upload-btn" type="button">📁 Elegir Archivo</button>
                    <input type="file" id="audioFile" accept=".wav,.mp3,.flac,.m4a,.ogg,.webm,.aac" onchange="handleFileSelect(this)" style="display: none;">
                </div>
                
                <div id="fileInfo" class="file-info" style="display: none;"></div>
                
                <button id="analyzeFileBtn" class="analyze-btn" disabled>
                    🔍 Analizar Archivo de Audio
                </button>
                
                <div class="troubleshooting">
                    <h4>🔧 Solución de problemas con archivos:</h4>
                    <ul>
                        <li><strong>Si el archivo no se procesa:</strong> Intenta convertirlo a WAV o MP3</li>
                        <li><strong>Para mejor compatibilidad:</strong> Usa archivos WAV de 16-bit, 22kHz</li>
                        <li><strong>Si persisten errores:</strong> Usa la grabación directa (más confiable)</li>
                        <li><strong>Formatos recomendados:</strong> WAV > MP3 > FLAC > otros</li>
                    </ul>
                </div>
            </div>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p id="loadingText">Procesando audio con múltiples métodos...</p>
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
        
        // Iniciar grabación
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 44100,
                        channelCount: 1,
                        volume: 1.0
                    } 
                });
                
                // Usar formato más compatible
                let mimeType = 'audio/webm';
                if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                    mimeType = 'audio/webm;codecs=opus';
                } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                    mimeType = 'audio/mp4';
                }
                
                mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = () => {
                    recordedBlob = new Blob(audioChunks, { type: mimeType });
                    const audioURL = URL.createObjectURL(recordedBlob);
                    document.getElementById('audioPlayer').src = audioURL;
                    document.getElementById('audioPlayerContainer').style.display = 'block';
                    document.getElementById('analyzeRecordingBtn').disabled = false;
                    
                    // Detener stream
                    stream.getTracks().forEach(track => track.stop());
                };
                
                // Iniciar grabación
                audioChunks = [];
                mediaRecorder.start(1000); // Grabar en chunks de 1 segundo
                
                // Timer
                recordingSeconds = 0;
                document.getElementById('recordingTimer').textContent = '00:00';
                document.getElementById('recordingTime').style.display = 'block';
                
                recordingTimer = setInterval(() => {
                    recordingSeconds++;
                    const minutes = Math.floor(recordingSeconds / 60);
                    const seconds = recordingSeconds % 60;
                    document.getElementById('recordingTimer').textContent = 
                        `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                    
                    // Limitar a 2 minutos
                    if (recordingSeconds >= 120) {
                        stopRecording();
                    }
                }, 1000);
                
                // UI
                document.getElementById('recordBtn').style.display = 'none';
                document.getElementById('stopBtn').style.display = 'inline-block';
                
            } catch (error) {
                console.error('Error accessing microphone:', error);
                showError('No se pudo acceder al micrófono. Asegúrate de dar permisos y usar HTTPS.');
            }
        }
        
        // Detener grabación
        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                clearInterval(recordingTimer);
                
                document.getElementById('recordBtn').style.display = 'inline-block';
                document.getElementById('stopBtn').style.display = 'none';
                document.getElementById('recordBtn').textContent = '🎙️ Grabar de nuevo';
            }
        }
        
        // Analizar grabación
        async function analyzeRecording() {
            if (!recordedBlob) {
                showError('No hay grabación para analizar');
                return;
            }
            
            showLoading();
            
            try {
                const formData = new FormData();
                formData.append('audio', recordedBlob, 'recording.webm');
                
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
                showError('Error al analizar: ' + error.message + '. Intenta grabar de nuevo con mejor calidad de audio.');
            }
        }
        
        // Manejar archivo
        function handleFileSelect(input) {
            const file = input.files[0];
            if (file) {
                handleFile(file);
            }
        }
        
        function handleFile(file) {
            const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/flac', 'audio/m4a', 'audio/ogg', 'audio/webm', 'audio/aac'];
            const fileType = file.type.toLowerCase();
            const fileName = file.name.toLowerCase();
            
            // Verificar por tipo MIME y extensión
            const isValidType = allowedTypes.some(type => fileType.includes(type.split('/')[1]));
            const isValidExtension = ['wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm', 'aac'].some(ext => fileName.endsWith('.' + ext));
            
            if (!isValidType && !isValidExtension) {
                showError('Formato no soportado. Usa: WAV, MP3, FLAC, M4A, OGG, WEBM, AAC');
                return;
            }
            
            if (file.size > 32 * 1024 * 1024) {
                showError('Archivo muy grande. Máximo 32MB.');
                return;
            }
            
            selectedFile = file;
            document.getElementById('fileInfo').style.display = 'block';
            document.getElementById('fileInfo').innerHTML = `
                <strong>📄 Archivo:</strong> ${file.name}<br>
                <strong>📏 Tamaño:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB<br>
                <strong>🎵 Tipo:</strong> ${file.type || 'Detectado por extensión'}<br>
                <strong>✅ Listo para análisis robusto</strong>
            `;
            document.getElementById('analyzeFileBtn').disabled = false;
            hideError();
        }
        
        // Analizar archivo
        async function analyzeFile() {
            if (!selectedFile) {
                showError('Selecciona un archivo primero');
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
                showError('Error: ' + error.message + '. Intenta con otro archivo o usa la grabación directa.');
            }
        }
        
        // Mostrar resultado
        function showResult(data) {
            const messages = {
                'Alegría': '🌟 ¡Qué maravilloso! Tu energía positiva es contagiosa. Sigue cultivando esos momentos de felicidad.',
                'Tristeza': '💙 Es normal sentir tristeza. Estos sentimientos son temporales. No estás solo.',
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
            
            const confidenceColor = data.confidence >= 80 ? '#27ae60' : data.confidence >= 70 ? '#f39c12' : '#e74c3c';
            
            document.getElementById('result').innerHTML = `
                <div class="result">
                    <div class="emotion">${emotionEmojis[data.emotion]} ${data.emotion}</div>
                    <div class="confidence" style="color: ${confidenceColor}">
                        Confianza: ${data.confidence}%
                    </div>
                    <div class="message">${messages[data.emotion]}</div>
                    <div class="features-info">
                        <strong>🔬 Análisis:</strong> ${data.features_analyzed} características procesadas<br>
                        <strong>⏱️ Duración:</strong> ${data.audio_duration?.toFixed(1) || 'N/A'}s<br>
                        <strong>📅 Procesado:</strong> ${new Date(data.timestamp).toLocaleString()}
                    </div>
                </div>
            `;
            document.getElementById('result').style.display = 'block';
            document.getElementById('result').scrollIntoView({ behavior: 'smooth' });
        }
        
        // Funciones de UI
        function showError(message) {
            const errorElement = document.getElementById('errorMessage');
            errorElement.textContent = message;
            errorElement.style.display = 'block';
        }
        
        function hideError() {
            document.getElementById('errorMessage').style.display = 'none';
        }
        
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('analyzeFileBtn').disabled = true;
            document.getElementById('analyzeRecordingBtn').disabled = true;
            hideError();
        }
        
        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('analyzeFileBtn').disabled = false;
            document.getElementById('analyzeRecordingBtn').disabled = false;
        }
        
        // Configurar pestañas
        function setupTabs() {
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    tabs.forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(content => {
                        content.classList.remove('active');
                    });
                    
                    tab.classList.add('active');
                    const tabId = tab.getAttribute('data-tab');
                    document.getElementById(tabId + 'Tab').classList.add('active');
                    
                    hideError();
                    document.getElementById('result').style.display = 'none';
                });
            });
        }
        
        // Inicializar
        document.addEventListener('DOMContentLoaded', () => {
            setupTabs();
            
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
            print(f"📁 Archivo guardado: {temp_path} ({os.path.getsize(temp_path)} bytes)")
            
            # Verificar que el archivo no esté vacío
            if os.path.getsize(temp_path) == 0:
                return jsonify({'error': 'El archivo de audio está vacío'}), 400
            
            # Extraer características con método robusto
            features = extract_audio_features_robust(temp_path)
            
            if features is None:
                return jsonify({'error': 'No se pudieron extraer características del audio. Intenta con un archivo diferente o graba directamente desde el navegador.'}), 500
            
            # Analizar emoción
            result = analyze_emotion_simple_robust(features)
            
            if result is None:
                return jsonify({'error': 'Error en el análisis emocional'}), 500
            
            print(f"✅ Análisis completado: {result['emotion']} ({result['confidence']}%)")
            
            response = {
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'features_analyzed': result['features_analyzed'],
                'audio_duration': result.get('audio_duration', 0),
                'timestamp': datetime.now().isoformat(),
                'filename': filename,
                'analysis_type': 'robust_processing'
            }
            
            return jsonify(response)
        
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                print(f"🗑️ Archivo temporal eliminado")
    
    except Exception as e:
        print(f"❌ Error en análisis: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}. Intenta con la grabación directa para mejores resultados.'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'mode': 'robust_analysis',
        'librosa_available': True,
        'max_file_size': '32MB',
        'supported_formats': ['WAV', 'MP3', 'FLAC', 'M4A', 'OGG', 'WEBM', 'AAC'],
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🧠 Asistente de Apoyo Emocional - VERSIÓN ROBUSTA")
    print("=" * 70)
    print("✅ Procesamiento robusto con múltiples métodos")
    print("✅ Mejor manejo de errores y compatibilidad")
    print("✅ Soporte extendido de formatos de audio")
    print("✅ Grabación optimizada desde navegador")
    print("=" * 70)
    print("🌐 Servidor: http://localhost:5000")
    print("📱 Interfaz web: http://localhost:5000")
    print("💚 Estado: http://localhost:5000/health")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
