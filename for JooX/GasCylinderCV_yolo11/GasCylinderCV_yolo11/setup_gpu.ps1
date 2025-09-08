
# setup_gpu.ps1 — Install CUDA PyTorch + deps in current venv (Windows)
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python numpy pyyaml matplotlib tqdm
