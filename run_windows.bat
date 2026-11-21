@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto :error
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :error

echo Installing base project dependencies...
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo Matching ML libraries to the Google Colab training environment...
python install_model_dependencies.py
if errorlevel 1 goto :error

echo Starting Streamlit...
python -m streamlit run app.py
goto :eof

:error
echo.
echo An error occurred. Check the messages above and verify that Python is installed and available in PATH.
pause
exit /b 1
