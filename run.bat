@echo off
echo 🧠 Asistente de Apoyo Emocional - Instalador
echo ==========================================

echo.
echo 📦 Actualizando pip...
python.exe -m pip install --upgrade pip

echo.
echo 📋 Instalando dependencias...
pip install -r requirements.txt

echo.
echo 📁 Creando carpetas necesarias...
if not exist "backend" mkdir backend
if not exist "backend\uploads" mkdir backend\uploads
if not exist "backend\models" mkdir backend\models

echo.
echo ✅ Instalación completada!
echo.
echo 🚀 Para ejecutar la aplicación:
echo    1. cd backend
echo    2. python app.py
echo    3. Abrir http://localhost:5000
echo.
pause
