@echo off
color 0A
echo ===============================================
echo    GAS CYLINDER DETECTION SYSTEM
echo    Ultra-Strict Mode (No False Positives)
echo ===============================================
echo.

REM Check for virtual environment in multiple locations
echo Checking for virtual environment...

if exist "..\venv\Scripts\activate.bat" (
    echo Found venv in parent directory
    call ..\venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Found venv in current directory
    call venv\Scripts\activate.bat
) else if exist "..\..\venv\Scripts\activate.bat" (
    echo Found venv two levels up
    call ..\..\venv\Scripts\activate.bat
) else (
    echo.
    echo WARNING: Virtual environment not found!
    echo Attempting to run with system Python...
    echo.
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import cv2, torch, ultralytics" >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Some packages might be missing!
    echo Installing required packages...
    pip install ultralytics torch torchvision opencv-python
    echo.
)

REM Check if model weights exist
if not exist "runs\train\cylinder_detector\weights\best.pt" (
    echo.
    echo WARNING: Trained model not found!
    echo Please train a model first using: python src\train_model.py
    echo Or attempting to use base YOLO model...
    echo.
    python src\ultra_strict_detector.py --weights yolo11n.pt --source 0 --conf 0.95
) else (
    echo Starting Ultra-Strict Detection with trained model...
    python src\ultra_strict_detector.py --weights runs\train\cylinder_detector\weights\best.pt --source 0 --conf 0.95
)

if errorlevel 1 (
    echo.
    echo ERROR: Detection failed!
    echo Please check the error messages above.
)

echo.
echo Press any key to exit...
pause >nul