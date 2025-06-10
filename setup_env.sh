#!/bin/bash

# Script to set up virtual environment for indoor navigation project
# Removes existing env, creates new one, installs dependencies, sets up SAM2 config and X11, and verifies setup

set -e  # Exit on any error

# Define variables
VENV_PATH="$HOME/sam2_env"
SAM2_CONFIG_SRC="/home/ekanshgupta92/sam2_configs/sam2_hiera_l.yaml"
SAM2_CONFIG_DIR="/home/ekanshgupta92/sam2_configs"
SAM2_CHECKPOINT="/home/ekanshgupta92/checkpoints/sam2_hiera_large.pth"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"

# Function to print messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Step 1: Remove existing virtual environment
log "Removing existing virtual environment at $VENV_PATH..."
if [ -d "$VENV_PATH" ]; then
    rm -rf "$VENV_PATH" || { log "ERROR: Failed to remove $VENV_PATH"; exit 1; }
fi

# Step 2: Create new virtual environment
log "Creating new virtual environment at $VENV_PATH..."
python3 -m venv "$VENV_PATH" || { log "ERROR: Failed to create virtual environment"; exit 1; }
source "$VENV_PATH/bin/activate" || { log "ERROR: Failed to activate virtual environment"; exit 1; }

# Step 3: Upgrade pip
log "Upgrading pip..."
pip install --upgrade pip || { log "ERROR: Failed to upgrade pip"; exit 1; }

# Step 4: Install dependencies
log "Installing dependencies..."
pip install numpy==1.24.4 || { log "ERROR: Failed to install numpy"; exit 1; }
pip install opencv-python==4.7.0.72 || { log "ERROR: Failed to install opencv-python"; exit 1; }
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 || { log "ERROR: Failed to install PyTorch"; exit 1; }
pip install transformers==4.35.2 timm==0.6.12 pyyaml==6.0 datasets==2.14.5 ai2thor hydra-core==1.3.2 accelerate==0.21.0 tqdm || { log "ERROR: Failed to install additional dependencies"; exit 1; }
pip install git+https://github.com/facebookresearch/segment-anything-2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4 || { log "ERROR: Failed to install SAM2"; exit 1; }

# Step 5: Copy SAM2 config file
log "Setting up SAM2 config file at $SAM2_CONFIG_SRC..."
if [ ! -f "$SAM2_CONFIG_SRC" ]; then
    log "Downloading SAM2 config file..."
    mkdir -p "$SAM2_CONFIG_DIR"
    wget https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2/configs/sam2/sam2_hiera_l.yaml -O "$SAM2_CONFIG_SRC" || { log "ERROR: Failed to download SAM2 config file"; exit 1; }
fi
chmod 644 "$SAM2_CONFIG_SRC" || { log "ERROR: Failed to set permissions on SAM2 config file"; exit 1; }

# Step 6: Download SAM2 checkpoint if missing
log "Checking SAM2 checkpoint at $SAM2_CHECKPOINT..."
if [ ! -f "$SAM2_CHECKPOINT" ]; then
    log "Downloading SAM2 checkpoint..."
    mkdir -p "$(dirname "$SAM2_CHECKPOINT")" || { log "ERROR: Failed to create checkpoint directory"; exit 1; }
    wget "$CHECKPOINT_URL" -O "$SAM2_CHECKPOINT" || { log "ERROR: Failed to download SAM2 checkpoint"; exit 1; }
fi

# Step 7: Set up X11 for AI2-THOR
log "Setting up X11 for AI2-THOR..."
sudo apt-get update || { log "ERROR: Failed to update apt"; exit 1; }
sudo apt-get install -y xvfb x11-xserver-utils libgl1-mesa-glx libgl1-mesa-dri xserver-xorg-core xserver-xorg-video-nvidia || { log "ERROR: Failed to install X11 dependencies"; exit 1; }

# Stop any existing Xvfb processes
pkill Xvfb || true
# Start Xvfb with reliable command
Xvfb :99 -screen 0 1024x768x24 &> /dev/null &
sleep 2  # Wait for Xvfb to start
export DISPLAY=:99
xdpyinfo &> /dev/null || { log "ERROR: X11 setup failed on :99"; exit 1; }

# Step 8: Set library path
log "Setting LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export DISPLAY=:99" >> ~/.bashrc

# Step 9: Verify environment
log "Verifying environment..."
python3 -c "import torch, zoedepth, transformers, cv2, sam2, ai2thor, numpy, hydra, accelerate, tqdm; print('All dependencies imported'); print(numpy.__version__, torch.__version__, hydra.__version__, accelerate.__version__)" || { log "ERROR: Failed to import dependencies"; exit 1; }
python3 -c "import hydra, os; hydra.core.global_hydra.GlobalHydra.instance().clear(); hydra.initialize_config_module('sam2_configs', version_base='1.2'); from sam2.build_sam import build_sam2; sam = build_sam2(config_file='sam2_hiera_l.yaml', checkpoint='/home/ekanshgupta92/checkpoints/sam2_hiera_large.pth', device='cuda', hydra_overrides_extra=['+searchpath=/home/ekanshgupta92/sam2_configs']); print('SAM2 loaded successfully')" || { log "ERROR: Failed to load SAM2"; exit 1; }
python3 -c "import ai2thor.controller; controller = ai2thor.controller.Controller(scene='FloorPlan1'); print('AI2-THOR loaded successfully'); controller.stop()" || { log "ERROR: Failed to load AI2-THOR"; exit 1; }

# Step 10: Verify CUDA
log "Verifying CUDA..."
nvcc --version || { log "ERROR: CUDA not found"; exit 1; }
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.backends.cudnn.enabled)" || { log "ERROR: CUDA verification failed"; exit 1; }

# Step 11: Verify dataset paths
log "Verifying dataset paths..."
[ -d "/home/ekanshgupta92/coco/train2017" ] || { log "ERROR: COCO train2017 directory not found"; exit 1; }
[ -f "/home/ekanshgupta92/coco/annotations/instances_train2017.json" ] || { log "ERROR: COCO annotations not found"; exit 1; }
[ -d "/home/ekanshgupta92/textvqa/images" ] || { log "ERROR: TextVQA images directory not found"; exit 1; }
[ -f "/home/ekanshgupta92/textvqa/textvqa.json" ] || { log "ERROR: TextVQA JSON not found"; exit 1; }

# Success message
log "Environment setup completed successfully!"
log "To use the environment, run: source ~/sam2_env/bin/activate"
log "Then run your script: python3 indoor_navigation_vlm.py"