#!/bin/bash

# CRCL Trading Dashboard Startup Script

echo "🚀 Starting CRCL Trading Dashboard..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "✅ Virtual environment found"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check dependencies
echo "📦 Checking dependencies..."
python -c "import flask, pandas, openpyxl; print('✅ All dependencies available')" || {
    echo "❌ Dependencies missing. Installing..."
    pip install -r requirements.txt
}

# Start the application
echo "🌐 Starting Flask application..."
echo "   Dashboard will be available at: http://localhost:5001"
echo "   Press Ctrl+C to stop the server"
echo "=================================="

python app.py
