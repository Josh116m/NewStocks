#!/bin/bash

echo "🚀 Weekly Stock Analysis System"
echo "================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python not found! Please install Python 3.8+ first."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "✅ Python found"
echo ""

# Run the analysis
echo "📊 Starting analysis..."
$PYTHON_CMD run_analysis.py

echo ""
echo "📋 Analysis complete! Check the output above."
echo "📁 Results saved to predictions/ folder"
