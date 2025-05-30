#!/bin/bash

# Quick conversion script for common models

echo "🚀 EmbedKit Model Converter"
echo "=========================="

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Default model
MODEL=${1:-"sentence-transformers/all-MiniLM-L6-v2"}

echo ""
echo "🔄 Converting model: $MODEL"
echo ""

# Run conversion
python3 convert_to_coreml.py "$MODEL" --output ./CoreMLModels --test

echo ""
echo "✅ Conversion complete!"
echo ""
echo "To use in EmbedKit:"
echo "1. Copy the .mlpackage file to your app bundle"
echo "2. Use the model identifier in your Swift code"
echo ""
echo "Example models to try:"
echo "  ./quick_convert.sh sentence-transformers/all-MiniLM-L6-v2    # 22MB, 384d"
echo "  ./quick_convert.sh BAAI/bge-small-en-v1.5                    # 33MB, 384d"
echo "  ./quick_convert.sh thenlper/gte-small                        # 33MB, 384d"