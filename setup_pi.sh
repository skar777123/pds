#!/bin/bash
# ==============================================================================
# One-Click Setup Script for Raspberry Pi 4B (64-bit Debian Bookworm)
# Pothole Detection System (PRD Implementation)
# ==============================================================================

set -e

echo "=========================================================="
echo "🚗 Setting up Pothole Detection System on Raspberry Pi 4B"
echo "=========================================================="

# 1. Update OS package index
echo "[1/6] Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-opencv \
    python3-numpy \
    v4l-utils \
    libatlas-base-dev \
    libjpeg-dev \
    zlib1g-dev \
    stress-ng

# 2. Configure USB webcam permissions
echo "[2/6] Configuring USB Webcam (V4L2) permissions..."
sudo usermod -a -G video $USER
sudo bash -c 'cat <<EOF > /etc/udev/rules.d/99-webcam.rules
SUBSYSTEM=="video4linux", GROUP="video", MODE="0660"
EOF'
sudo udevadm control --reload-rules && sudo udevadm trigger

# 3. Create virtual environment with system site-packages (links system OpenCV)
echo "[3/6] Setting up Python virtual environment..."
VENV_DIR="$HOME/pds_venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
    echo "Virtual environment created at: $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 4. Install production Python packages
echo "[4/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements-pi.txt

# 5. Create log directories
echo "[5/6] Creating log directories..."
sudo mkdir -p /var/log/pothole_system
sudo chown -R $USER:$USER /var/log/pothole_system
mkdir -p logs

# 6. Verify webcam connectivity
echo "[6/6] Checking Portronics Webcam connection..."
if v4l2-ctl --list-devices | grep -q "/dev/video"; then
    echo "✅ Portronics USB Webcam detected on V4L2 bus!"
else
    echo "⚠️ Warning: No USB webcam detected on /dev/video*. Please plug in your Portronics camera."
fi

echo "=========================================================="
echo "🎉 Setup Complete!"
echo "To test camera angle & ROI calibration: python scripts/test_camera.py"
echo "To run live edge pothole detection:     python edge/detect_edge.py"
echo "=========================================================="
