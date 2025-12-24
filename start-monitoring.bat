@echo off
REM Job Hunter AI - Monitoring Quick Start Script (Windows)
REM This script sets up and starts the complete monitoring stack

echo.
echo Job Hunter AI - Monitoring Stack Setup
echo ==========================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo Docker Compose is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

echo Docker and Docker Compose found
echo.

REM Check for .env file
if not exist .env (
    echo .env file not found. Creating template...
    (
        echo # Required API Keys
        echo OPENAI_API_KEY=your-openai-key-here
        echo RAPIDAPI_KEY=your-rapidapi-key-here
        echo.
        echo # Optional: Custom Ports
        echo STREAMLIT_PORT=8501
        echo METRICS_PORT=8000
        echo PROMETHEUS_PORT=9090
        echo GRAFANA_PORT=3000
        echo ALERTMANAGER_PORT=9093
        echo.
        echo # Grafana Credentials
        echo GRAFANA_ADMIN_USER=admin
        echo GRAFANA_ADMIN_PASSWORD=admin123
    ) > .env
    echo Created .env template. Please edit it with your API keys.
    echo Then run this script again.
    pause
    exit /b 1
)

echo .env file found
echo.

REM Create necessary directories
echo Creating directories...
if not exist monitoring\grafana\dashboards mkdir monitoring\grafana\dashboards
if not exist monitoring\grafana\datasources mkdir monitoring\grafana\datasources
if not exist output\resumes mkdir output\resumes
if not exist uploads mkdir uploads
if not exist logs mkdir logs

echo Directories created
echo.

REM Build Docker images
echo Building Docker images...
docker-compose -f docker-compose.monitoring.yml build

echo Images built
echo.

REM Start services
echo Starting monitoring stack...
docker-compose -f docker-compose.monitoring.yml up -d

echo.
echo Waiting for services to be healthy...
timeout /t 10 /nobreak >nul

REM Check service health
echo.
echo Checking service health...
docker-compose -f docker-compose.monitoring.yml ps

echo.
echo Monitoring Stack Started!
echo.
echo Access Points:
echo ================================
echo   Streamlit App:  http://localhost:8501
echo   Metrics:        http://localhost:8000/metrics
echo   Prometheus:     http://localhost:9090
echo   Grafana:        http://localhost:3000
echo                   (admin / admin123)
echo   AlertManager:   http://localhost:9093
echo ================================
echo.
echo Next Steps:
echo   1. Visit Streamlit app and start using Job Hunter AI
echo   2. Check Grafana dashboard for real-time metrics
echo   3. Configure alerts in monitoring\alertmanager.yml
echo.
echo Documentation:
echo   - Setup Guide:       monitoring\README.md
echo   - Integration Guide: monitoring\INTEGRATION.md
echo.
echo To stop: docker-compose -f docker-compose.monitoring.yml down
echo To restart: docker-compose -f docker-compose.monitoring.yml restart
echo To view logs: docker-compose -f docker-compose.monitoring.yml logs -f
echo.
echo Happy Job Hunting!
echo.
pause
