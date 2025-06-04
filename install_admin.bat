@echo off
echo 🔧 Instalador con permisos de administrador
echo ==========================================

echo.
echo ⚠️  Este script requiere permisos de administrador
echo    Haz clic derecho y selecciona "Ejecutar como administrador"
echo.

REM Verificar si se está ejecutando como administrador
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Ejecutándose con permisos de administrador
) else (
    echo ❌ Se requieren permisos de administrador
    echo    Haz clic derecho en este archivo y selecciona "Ejecutar como administrador"
    pause
    exit /b 1
)

echo.
echo 📦 Actualizando pip...
python.exe -m pip install --upgrade pip

echo.
echo 📋 Instalando dependencias básicas...
pip install Flask==3.0.3
pip install Flask-CORS==4.0.0
pip install numpy==1.24.3
pip install werkzeug==3.0.3
pip install python-dotenv==1.0.0

echo.
echo 🎵 Instalando dependencias de audio...
pip install soundfile
pip install librosa==0.10.1

echo.
echo ✅ Instalación completada!
echo.
echo 🚀 Para ejecutar:
echo    python app.py
echo.
pause
