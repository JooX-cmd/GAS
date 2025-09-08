@echo off
echo Starting Ultra-Strict Gas Cylinder Detection System...
echo This mode will ONLY detect tall, centered objects (real gas cylinders)
echo.

REM Activate virtual environment
call ..\venv\Scripts\activate.bat

REM Run the ultra-strict detection system
python src\ultra_strict_detector.py --weights runs\train\cylinder_detector\weights\best.pt --source 0 --conf 0.95

pause
