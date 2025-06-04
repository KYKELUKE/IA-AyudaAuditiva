#!/usr/bin/env python3
"""
Script para verificar que todas las dependencias estén instaladas correctamente
"""

import sys
import importlib
import subprocess

def check_python_version():
    """Verificar versión de Python"""
    version = sys.version_info
    print(f"🐍 Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Se requiere Python 3.8 o superior")
        return False
    else:
        print("✅ Versión de Python compatible")
        return True

def check_package(package_name, import_name=None):
    """Verificar si un paquete está instalado"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Desconocida')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: NO INSTALADO")
        return False

def check_ffmpeg():
    """Verificar si FFmpeg está disponible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg: {version_line}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        print("⚠️ FFmpeg: NO DISPONIBLE (opcional)")
        return False

def main():
    print("🔍 Verificando dependencias del Asistente de Apoyo Emocional")
    print("=" * 60)
    
    all_good = True
    
    # Verificar Python
    if not check_python_version():
        all_good = False
    
    print("\n📦 Verificando paquetes de Python:")
    
    # Paquetes requeridos
    required_packages = [
        ('Flask', 'flask'),
        ('Flask-CORS', 'flask_cors'),
        ('librosa', 'librosa'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('scikit-learn', 'sklearn'),
        ('werkzeug', 'werkzeug'),
        ('python-dotenv', 'dotenv')
    ]
    
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_good = False
    
    print("\n🔧 Verificando herramientas opcionales:")
    check_ffmpeg()
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("🎉 ¡Todas las dependencias principales están instaladas!")
        print("\n🚀 Para ejecutar la aplicación:")
        print("   python app_robust.py")
        print("\n🌐 Luego abre: http://localhost:5000")
    else:
        print("❌ Faltan algunas dependencias.")
        print("\n💡 Para instalar las dependencias faltantes:")
        print("   pip install flask flask-cors librosa numpy scipy scikit-learn werkzeug python-dotenv")
    
    print("\n📋 Información del sistema:")
    print(f"   Sistema operativo: {sys.platform}")
    print(f"   Arquitectura: {sys.maxsize > 2**32 and '64-bit' or '32-bit'}")

if __name__ == "__main__":
    main()
