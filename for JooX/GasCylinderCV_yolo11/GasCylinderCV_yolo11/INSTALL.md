# Gas Cylinder Detection System Installation Guide

## Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional but recommended)
- Git

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/JooX-cmd/GAS.git
cd GAS
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python src/train_model.py --help
```

## Usage

### Training
```bash
python src/train_model.py --data data/dataset/data.yaml --epochs 50
```

### Testing
```bash
python src/test_model.py --weights runs/train/cylinder_detector/weights/best.pt
```

### Running Detection
```bash
python src/ultra_strict_detector.py --weights runs/train/cylinder_detector/weights/best.pt
```

## Troubleshooting



For other issues, please check the error logs or raise an issue on GitHub.