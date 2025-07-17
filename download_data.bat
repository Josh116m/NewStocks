@echo off
echo 📡 Fresh Data Downloader
echo ========================

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

echo 📥 Downloading fresh stock data...
echo ⏱️  This will take 10-20 minutes
echo 🔑 API credentials included - no setup needed
echo.

REM Run the data downloader
python download_fresh_data.py

echo.
echo 📋 Download complete! Check the output above.
echo 🚀 You can now run: python weekly_analysis.py
echo.
pause
