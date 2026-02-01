#!/bin/bash
set -e # หยุดทำงานทันทีถ้ามี error เกิดขึ้น

# Configuration
PROJECT_NAME="inventory_forecasting"
TARGET_DIR="$HOME/projects/$PROJECT_NAME"
SOURCE_DIR=$(pwd)

echo "========================================================"
echo "🚀 Starting Migration: Windows -> Linux Native"
echo "📍 Source: $SOURCE_DIR"
echo "🎯 Target: $TARGET_DIR"
echo "🎮 GPU Check: NVIDIA GeForce RTX 4060 (Estimated)"
echo "========================================================"

# 0. Safety Check
if [ ! -f "$SOURCE_DIR/requirements.txt" ]; then
    echo "❌ Error: ไม่พบไฟล์ requirements.txt ในโฟลเดอร์ปัจจุบัน!"
    echo "   โปรดรัน script นี้จาก root folder ของโปรเจกต์ต้นทาง"
    exit 1
fi

# 1. GPU Driver Check
echo "🔍 Checking for NVIDIA Drivers..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA Drivers found:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  WARNING: ไม่พบคำสั่ง nvidia-smi!"
    echo "   กรุณาติดตั้ง NVIDIA Drivers ก่อนเพื่อให้ TensorFlow ใช้ GPU ได้"
    echo "   (หรือกด Ctrl+C เพื่อยกเลิกตอนนี้ แล้วไปลง driver ก่อน)"
    sleep 5
fi

# 2. Install System Dependencies
echo "📦 Installing System Dependencies..."
# เพิ่ม python3-dev เพื่อป้องกัน error เวลา compile library บางตัว
sudo apt update && sudo apt install -y python3-full python3-pip python3-venv python3-dev git

# 3. Create Target & Copy Files
echo "📂 Setting up target directory..."
mkdir -p "$TARGET_DIR"

echo "🚚 Copying files using rsync..."
# เอา --exclude '.git' ออก เพื่อให้ Git ทำงานต่อได้
# เพิ่ม --delete เพื่อให้ปลายทางเหมือนต้นทางเป๊ะๆ (ระวังไฟล์ปลายทางหายถ้าไม่มีในต้นทาง)
rsync -av --progress "$SOURCE_DIR/" "$TARGET_DIR/" \
    --exclude 'venv' \
    --exclude 'venv_wsl' \
    --exclude '.env' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '.idea' \
    --exclude '.vscode' 
    # หมายเหตุ: ผม exclude .env เผื่อไว้ เพราะบางที config ของ windows/linux ต่างกัน แต่ถ้าเหมือนกันก็เอาออกได้ครับ

echo "✅ Copy complete."

# 4. Setup Python Environment
cd "$TARGET_DIR"
echo "🐍 Setting up venv at $(pwd)..."

# ลบ venv เก่าทิ้งก่อนเสมอเพื่อความชัวร์
rm -rf venv 
python3 -m venv venv

source venv/bin/activate
pip install --upgrade pip setuptools wheel

# 5. Install Python Libraries
echo "⬇️ Installing libraries..."

# Option A: ลง TF GPU ก่อน เพื่อความชัวร์ (แนะนำสำหรับ RTX 40xx)
pip install "tensorflow[and-cuda]" 

# Option B: ลงตัวที่เหลือตาม requirements
if [ -f "requirements.txt" ]; then
    # Exclude tensorflow from requirements if it creates conflict, or just install over it
    pip install -r requirements.txt
fi

echo ""
echo "🎉 MIGRATION SUCCESSFUL! 🎉"
echo "========================================================"
echo "To start working:"
echo "  cd $TARGET_DIR"
echo "  source venv/bin/activate"
echo "  python3 run_batch_experiment.py"
echo "========================================================"