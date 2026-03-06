@echo off
echo ========================================================
echo   AquaSentinel - Krishna River Pollution Intelligence
echo ========================================================
echo.
echo [1] Checking dependencies...
pip install -r requirements.txt
echo.
echo [2] Launching Streamlit Dashboard...
streamlit run dashboard/command_center.py
echo.
pause
