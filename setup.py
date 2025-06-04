import os
import subprocess
import sys

def install_dependencies():
    """
    Instala todas las dependencias necesarias para el proyecto
    """
    dependencies = [
        'Flask==3.0.3',
        'Flask-CORS==4.0.0',
        'librosa==0.10.1',
        'numpy==1.24.3',
        'scipy==1.11.1',
        'scikit-learn==1.3.0',
        'werkzeug==3.0.3',
        'python-dotenv==1.0.0',
        'joblib==1.3.2'
    ]
    
    print("🚀 Instalando dependencias para el Asistente de Apoyo Emocional...")
    print("=" * 60)
    
    for package in dependencies:
        try:
            print(f"📦 Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} instalado correctamente")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando {package}: {e}")
            return False
    
    print("\n🎉 ¡Todas las dependencias instaladas correctamente!")
    return True

def create_project_structure():
    """
    Crea la estructura de carpetas del proyecto
    """
    folders = [
        'backend',
        'backend/models',
        'backend/uploads',
        'frontend'
    ]
    
    print("\n📁 Creando estructura de carpetas...")
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Carpeta creada: {folder}")

def create_requirements_file():
    """
    Crea el archivo requirements.txt
    """
    requirements_content = """Flask==3.0.3
Flask-CORS==4.0.0
librosa==0.10.1
numpy==1.24.3
scipy==1.11.1
scikit-learn==1.3.0
werkzeug==3.0.3
python-dotenv==1.0.0
joblib==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
pandas==2.0.3
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    
    print("✅ Archivo requirements.txt creado")

if __name__ == "__main__":
    print("🧠 Configurando Asistente de Apoyo Emocional")
    print("=" * 50)
    
    # Crear estructura del proyecto
    create_project_structure()
    
    # Crear archivo requirements.txt
    create_requirements_file()
    
    # Instalar dependencias
    success = install_dependencies()
    
    if success:
        print("\n🎯 Configuración completada!")
        print("\nPróximos pasos:")
        print("1. cd backend")
        print("2. python app.py")
        print("3. Abrir http://localhost:5000 en tu navegador")
    else:
        print("\n❌ Hubo errores en la instalación. Revisa los mensajes anteriores.")
