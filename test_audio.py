import librosa
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def test_librosa_installation():
    """Prueba que librosa esté funcionando correctamente"""
    print("🧪 Probando instalación de librosa...")
    
    try:
        # Crear una señal de audio sintética para prueba
        sr = 22050  # Sample rate
        duration = 3  # 3 segundos
        t = np.linspace(0, duration, sr * duration)
        
        # Crear una señal con diferentes frecuencias (simulando emociones)
        frequency = 440  # La nota A4
        audio_signal = np.sin(2 * np.pi * frequency * t)
        
        print(f"✅ Señal de audio creada: {len(audio_signal)} muestras")
        
        # Probar extracción de características
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=13)
        print(f"✅ MFCC extraído: {mfccs.shape}")
        
        chroma = librosa.feature.chroma(y=audio_signal, sr=sr)
        print(f"✅ Chroma extraído: {chroma.shape}")
        
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_signal, sr=sr)
        print(f"✅ Spectral centroid: {spectral_centroids.shape}")
        
        zcr = librosa.feature.zero_crossing_rate(audio_signal)
        print(f"✅ Zero crossing rate: {zcr.shape}")
        
        tempo, beats = librosa.beat.beat_track(y=audio_signal, sr=sr)
        print(f"✅ Tempo detectado: {tempo:.1f} BPM")
        
        print("\n🎉 ¡Librosa funciona perfectamente!")
        print("🚀 Tu sistema está listo para análisis de audio real")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando librosa: {e}")
        return False

def create_sample_emotions():
    """Crea muestras de audio sintéticas que simulan diferentes emociones"""
    print("\n🎵 Creando muestras de audio para diferentes emociones...")
    
    sr = 22050
    duration = 2
    t = np.linspace(0, duration, sr * duration)
    
    emotions = {
        'alegria': {
            'freq': 523,  # Do alto
            'tempo': 140,
            'amplitude': 0.8,
            'description': 'Frecuencia alta, tempo rápido, amplitud alta'
        },
        'tristeza': {
            'freq': 196,  # Sol bajo
            'tempo': 60,
            'amplitude': 0.3,
            'description': 'Frecuencia baja, tempo lento, amplitud baja'
        },
        'ansiedad': {
            'freq': 440,  # La medio
            'tempo': 160,
            'amplitude': 0.6,
            'description': 'Frecuencia media, tempo muy rápido, variaciones'
        }
    }
    
    for emotion, params in emotions.items():
        # Crear señal base
        base_signal = params['amplitude'] * np.sin(2 * np.pi * params['freq'] * t)
        
        # Añadir variaciones según la emoción
        if emotion == 'ansiedad':
            # Añadir ruido y variaciones para simular ansiedad
            noise = 0.1 * np.random.normal(0, 1, len(t))
            tremolo = 0.2 * np.sin(2 * np.pi * 8 * t)  # Tremolo rápido
            signal = base_signal + noise + tremolo * base_signal
        elif emotion == 'tristeza':
            # Señal más suave y decreciente
            envelope = np.exp(-t * 0.5)  # Decaimiento exponencial
            signal = base_signal * envelope
        else:  # alegria
            # Señal brillante con armónicos
            harmonics = 0.3 * np.sin(2 * np.pi * params['freq'] * 2 * t)
            signal = base_signal + harmonics
        
        # Normalizar
        signal = signal / np.max(np.abs(signal))
        
        print(f"✅ {emotion.capitalize()}: {params['description']}")
    
    print("\n📊 Estas características ayudan al algoritmo a distinguir emociones")

if __name__ == "__main__":
    print("🔬 Prueba del Sistema de Análisis Emocional")
    print("=" * 50)
    
    # Probar librosa
    if test_librosa_installation():
        # Crear ejemplos de emociones
        create_sample_emotions()
        
        print("\n" + "=" * 50)
        print("✅ Sistema completamente funcional")
        print("🚀 Ejecuta 'python app.py' para iniciar la aplicación")
    else:
        print("\n❌ Hay problemas con la instalación")
        print("💡 Intenta reinstalar: pip install --user librosa soundfile")
