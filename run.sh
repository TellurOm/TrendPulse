#!/bin/bash
# One-click startup script for AIML Major Project - Case Study 127
cd "$(dirname "$0")"

echo "=========================================================="
echo "⚡ Starting TrendPulse: Social Media Trend ML Framework"
echo "   AIML Major Project - Case Study No. 127"
echo "=========================================================="

if [ -d "venv" ]; then
    source venv/bin/activate
fi

python3 -m streamlit run app.py --server.headless=true
