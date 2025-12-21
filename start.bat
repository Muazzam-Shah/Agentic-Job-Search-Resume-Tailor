@echo off
REM Job Hunter Web Application Quick Start Script (Windows)

echo ================================================
echo   Job Hunter - AI-Powered Resume Optimizer
echo ================================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.12+
    pause
    exit /b 1
)

echo ✓ Python found
echo.

REM Check for .env file
if not exist ".env" (
    echo ⚠️  No .env file found. Creating template...
    (
        echo # OpenAI API ^(Required^)
        echo OPENAI_API_KEY=your_openai_api_key_here
        echo.
        echo # Job APIs ^(At least one required^)
        echo ADZUNA_APP_ID=your_adzuna_app_id
        echo ADZUNA_API_KEY=your_adzuna_api_key
        echo.
        echo # Optional APIs
        echo GITHUB_TOKEN=your_github_token
        echo GOOGLE_API_KEY=your_google_api_key
        echo GOOGLE_CSE_ID=your_google_cse_id
    ) > .env
    echo ✓ Created .env template. Please add your API keys.
    echo.
    echo To get API keys:
    echo   - OpenAI: https://platform.openai.com/api-keys
    echo   - Adzuna: https://developer.adzuna.com/
    echo.
    pause
    exit /b 1
)

echo ✓ .env file found
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✓ Dependencies installed
echo.

REM Create necessary directories
echo Creating directories...
if not exist "uploads" mkdir uploads
if not exist "output" mkdir output
if not exist "data\chromadb" mkdir data\chromadb
if not exist "logs" mkdir logs

echo ✓ Directories created
echo.

REM Start Flask application
echo ================================================
echo   Starting Flask Application
echo ================================================
echo.
echo The application will be available at:
echo   🌐 http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py
