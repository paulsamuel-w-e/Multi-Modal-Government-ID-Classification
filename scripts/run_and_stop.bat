@echo off
REM === Launch FastAPI in Terminal 1 ===
start "FastAPI Server" cmd /k "call env\hvenv\Scripts\activate.bat && uvicorn main_fastapi:app --reload --port 8000"

REM === Launch Streamlit in Terminal 2 ===
start "Streamlit App" cmd /k "call env\hvenv\Scripts\activate.bat && streamlit run streamlit_ui\app.py"

REM === Ask User to Stop Servers ===
:wait_input
echo.
set /p userinput= Do you want to stop both servers? (y/n): 

if /I "%userinput%"=="y" (
    echo Stopping servers...

    REM === Kill Streamlit ===
    taskkill /F /IM streamlit.exe >nul 2>&1

    REM === Kill FastAPI (uvicorn usually runs under python.exe) ===
    taskkill /F /FI "WINDOWTITLE eq FastAPI Server" >nul 2>&1
    taskkill /F /FI "WINDOWTITLE eq Streamlit App" >nul 2>&1

    echo Servers stopped and terminals closed.
    exit
) else if /I "%userinput%"=="n" (
    echo Servers will continue running...
    goto wait_input
) else (
    echo Invalid input. Please enter y or n.
    goto wait_input
)