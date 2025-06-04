import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class EmotionAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.emotions = ['Alegría', 'Tristeza', 'Ansiedad', 'Neutral', 'Enojo']
        
    def extract_features(self, audio_path):
        """
        Extrae características avanzadas del audio
        """
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=30)
            
            features = []
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend([np.mean(mfccs), np.std(mfccs)])
            features.extend(np.mean(mfccs, axis=1))
            features.extend(np.std(mfccs, axis=1))
            
            # Chroma features
            chroma = librosa.feature.chroma(y=y, sr=sr)
            features.extend([np.mean(chroma), np.std(chroma)])
            features.extend(np.mean(chroma, axis=1))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.append(np.mean(spectral_centroids))
            features.append(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.append(np.mean(spectral_rolloff))
            features.append(np.std(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features.append(np.mean(spectral_bandwidth))
            features.append(np.std(spectral_bandwidth))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)
            
            # RMS Energy
            rms = librosa.feature.rms(y=y)
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def train_model(self, training_data_path):
        """
        Entrena el modelo con datos de entrenamiento
        """
        # Aquí cargarías tus datos de entrenamiento
        # Por ahora, creamos datos sintéticos para demostración
        
        print("Entrenando modelo de análisis emocional...")
        
        # Datos sintéticos para demostración
        X_train = np.random.rand(1000, 50)  # 1000 muestras, 50 características
        y_train = np.random.randint(0, 5, 1000)  # 5 emociones
        
        # Entrenar el modelo
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Normalizar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Entrenar
        self.model.fit(X_train_scaled, y_train)
        
        print("Modelo entrenado exitosamente!")
        
        # Guardar modelo
        self.save_model()
    
    def predict_emotion(self, audio_path):
        """
        Predice la emoción del archivo de audio
        """
        if self.model is None:
            self.load_model()
        
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
        # Normalizar características
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predicción
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        emotion = self.emotions[prediction]
        confidence = int(max(probabilities) * 100)
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': {
                self.emotions[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
    
    def save_model(self):
        """
        Guarda el modelo entrenado
        """
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/emotion_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("Modelo guardado en models/")
    
    def load_model(self):
        """
        Carga el modelo pre-entrenado
        """
        try:
            self.model = joblib.load('models/emotion_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            print("Modelo cargado exitosamente!")
        except FileNotFoundError:
            print("No se encontró modelo entrenado. Entrenando nuevo modelo...")
            self.train_model(None)

# Ejemplo de uso
if __name__ == "__main__":
    analyzer = EmotionAnalyzer()
    
    # Entrenar modelo (solo la primera vez)
    # analyzer.train_model(None)
    
    # Cargar modelo existente
    analyzer.load_model()
    
    print("Analizador de emociones listo!")
