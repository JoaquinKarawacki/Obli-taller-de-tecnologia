@echo off
rem Abre la demo del video en el navegador, con el entorno del proyecto.
cd /d "%~dp0"
echo Abriendo Jupyter... no cierres esta ventana mientras grabas.
"venv\Scripts\jupyter-notebook.exe" demo_video.ipynb
pause
