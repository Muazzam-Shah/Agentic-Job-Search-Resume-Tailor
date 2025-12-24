@echo off
echo ========================================
echo   Job Hunter AI - Conversational Agent
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Error: Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if .env exists
if not exist ".env" (
    echo.
    echo Warning: .env file not found!
    echo Please create .env with your API keys:
    echo   OPENAI_API_KEY=your-key-here
    echo   JSEARCH_API_KEY=your-key-here
    echo   ADZUNA_APP_ID=your-id-here
    echo   ADZUNA_APP_KEY=your-key-here
    echo.
    pause
    exit /b 1
)

REM Install/update dependencies
echo.
echo Installing dependencies...
pip install -q -r requirements.txt

REM Start the chat application
echo.
echo ========================================
echo   Starting Job Hunter Chat...
echo ========================================
echo.
echo Chat Interface: http://localhost:8501
echo.
echo Press CTRL+C to stop the server
echo.

streamlit run app_streamlit.py

pause
