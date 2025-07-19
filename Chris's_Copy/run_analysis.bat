@echo off
echo 🚀 Weekly Stock Analysis System
echo ================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Run the analysis
echo 📊 Starting analysis...
python run_analysis.py

echo.
echo 📋 Analysis complete! Check the output above.
echo 📁 Results saved to predictions/ folder
echo.
pause
