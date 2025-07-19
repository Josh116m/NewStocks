@echo off
echo 🚀 DAILY DATA UPDATE
echo ==================
echo.
echo 📅 Updating stock data to latest available...
echo.

cd /d "%~dp0"
python daily_data_update.py

echo.
echo 💡 To run predictions with updated data:
echo    cd .. 
echo    python weekly_analysis.py
echo.
pause
