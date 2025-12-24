#!/bin/bash

# Job Hunter AI - Monitoring Quick Start Script
# This script sets up and starts the complete monitoring stack

set -e

echo "🚀 Job Hunter AI - Monitoring Stack Setup"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose found"
echo ""

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating template..."
    cat > .env << EOF
# Required API Keys
OPENAI_API_KEY=your-openai-key-here
RAPIDAPI_KEY=your-rapidapi-key-here

# Optional: Custom Ports
STREAMLIT_PORT=8501
METRICS_PORT=8000
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
ALERTMANAGER_PORT=9093

# Grafana Credentials
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin123
EOF
    echo "📝 Created .env template. Please edit it with your API keys."
    echo "   Then run this script again."
    exit 1
fi

echo "✅ .env file found"
echo ""

# Check if API keys are set
source .env
if [ "$OPENAI_API_KEY" = "your-openai-key-here" ] || [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Please set your OPENAI_API_KEY in .env file"
    exit 1
fi

if [ "$RAPIDAPI_KEY" = "your-rapidapi-key-here" ] || [ -z "$RAPIDAPI_KEY" ]; then
    echo "❌ Please set your RAPIDAPI_KEY in .env file"
    exit 1
fi

echo "✅ API keys configured"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
mkdir -p output/resumes
mkdir -p uploads
mkdir -p logs

echo "✅ Directories created"
echo ""

# Build Docker images
echo "🔨 Building Docker images..."
docker-compose -f docker-compose.monitoring.yml build

echo "✅ Images built"
echo ""

# Start services
echo "🚀 Starting monitoring stack..."
docker-compose -f docker-compose.monitoring.yml up -d

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check service health
echo ""
echo "🏥 Checking service health..."
docker-compose -f docker-compose.monitoring.yml ps

echo ""
echo "✅ Monitoring Stack Started!"
echo ""
echo "📊 Access Points:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Streamlit App:  http://localhost:${STREAMLIT_PORT:-8501}"
echo "  Metrics:        http://localhost:${METRICS_PORT:-8000}/metrics"
echo "  Prometheus:     http://localhost:${PROMETHEUS_PORT:-9090}"
echo "  Grafana:        http://localhost:${GRAFANA_PORT:-3000}"
echo "                  (admin / ${GRAFANA_ADMIN_PASSWORD:-admin123})"
echo "  AlertManager:   http://localhost:${ALERTMANAGER_PORT:-9093}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 Next Steps:"
echo "  1. Visit Streamlit app and start using Job Hunter AI"
echo "  2. Check Grafana dashboard for real-time metrics"
echo "  3. Configure alerts in monitoring/alertmanager.yml"
echo ""
echo "📚 Documentation:"
echo "  - Setup Guide:       monitoring/README.md"
echo "  - Integration Guide: monitoring/INTEGRATION.md"
echo ""
echo "🛑 To stop: docker-compose -f docker-compose.monitoring.yml down"
echo "🔄 To restart: docker-compose -f docker-compose.monitoring.yml restart"
echo "📋 To view logs: docker-compose -f docker-compose.monitoring.yml logs -f"
echo ""
echo "Happy Job Hunting! 🎯"
