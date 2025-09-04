#!/bin/bash

echo "=================================================="
echo "Advanced Multi-Agent System Setup"
echo "=================================================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama is not installed!"
    echo ""
    echo "Please install Ollama first:"
    echo "  macOS:  brew install ollama"
    echo "  Linux:  curl -fsSL https://ollama.ai/install.sh | sh"
    echo ""
    exit 1
fi

echo "✅ Ollama is installed"
echo ""

# Start Ollama service if not running
echo "Starting Ollama service..."
ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!
sleep 2

# Small models for limited hardware
echo "📦 Pulling small models for limited hardware..."
echo ""

MODELS=(
    "huihui_ai/qwen3-abliterated:1.7b"
    "huihui_ai/llama3.2-abliterate:1b"
    "huihui_ai/falcon3-abliterated:1b"
    "huihui_ai/gemma3-abliterated:1b"
    "huihui_ai/qwen3-abliterated:0.6b"
)

for model in "${MODELS[@]}"; do
    echo "Pulling $model..."
    ollama pull "$model"
    if [ $? -eq 0 ]; then
        echo "  ✅ $model pulled successfully"
    else
        echo "  ❌ Failed to pull $model"
    fi
    echo ""
done

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "✨ Setup Complete!"
echo "=================================================="
echo ""
echo "To run the system:"
echo "  python3 quickstart.py"
echo ""
echo "Available models:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""
