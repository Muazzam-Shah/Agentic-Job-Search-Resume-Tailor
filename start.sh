#!/bin/bash

# Job Hunter Web Application Quick Start Script

echo "================================================"
echo "  Job Hunter - AI-Powered Resume Optimizer"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python --version || python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python not found. Please install Python 3.12+"
    exit 1
fi

echo "✓ Python found"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating template..."
    cat > .env << EOF
# OpenAI API (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Job APIs (At least one required)
ADZUNA_APP_ID=your_adzuna_app_id
ADZUNA_API_KEY=your_adzuna_api_key

# Optional APIs
GITHUB_TOKEN=your_github_token
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
EOF
    echo "✓ Created .env template. Please add your API keys."
    echo ""
    echo "To get API keys:"
    echo "  - OpenAI: https://platform.openai.com/api-keys"
    echo "  - Adzuna: https://developer.adzuna.com/"
    echo ""
    exit 1
fi

echo "✓ .env file found"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads output data/chromadb logs

echo "✓ Directories created"
echo ""

# Start Flask application
echo "================================================"
echo "  Starting Flask Application"
echo "================================================"
echo ""
echo "The application will be available at:"
echo "  🌐 http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
