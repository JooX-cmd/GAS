# CLAUDE.md - Gas Cylinder Detection System

## Project Overview
This is a YOLOv11-based gas cylinder detection system for computer vision applications.

## Project Structure
```
GasCylinderCV_yolo11/
├── data/dataset/          # Training and test datasets
├── runs/                  # Training and detection results
├── src/                   # Source code
│   ├── train_model.py          # Model training script
│   ├── test_single_image.py    # Single image testing script
│   └── ultra_strict_detector.py # Strict detection variant
├── BACKUP_SUMMARY.md      # Project backup summary
├── INSTALL.md             # Installation instructions
├── PROJECT_GUIDE.md       # Detailed project documentation
├── README.md              # Project README
├── requirements.txt       # Python dependencies
├── run_ultra_strict.bat   # Batch script for ultra strict detection
├── SETUP_AFTER_RESET.bat  # Setup script after reset
└── yolo11n.pt            # Pre-trained YOLOv11 model
```

## Key Commands

### Training
```bash
cd "for JooX/GasCylinderCV_yolo11/GasCylinderCV_yolo11"
python src/train_model.py
```

### Single Image Testing
```bash
cd "for JooX/GasCylinderCV_yolo11/GasCylinderCV_yolo11"
python src/test_single_image.py
```

### Ultra Strict Detection
```bash
cd "for JooX/GasCylinderCV_yolo11/GasCylinderCV_yolo11"
python src/ultra_strict_detector.py
```

### Using Batch Script
```bash
cd "for JooX/GasCylinderCV_yolo11/GasCylinderCV_yolo11"
./run_ultra_strict.bat
```

## Dependencies
See `requirements.txt` for complete list. Key dependencies include:
- torch>=2.0.0 (with CUDA support)
- torchvision>=0.15.0
- ultralytics>=8.0.0 (YOLOv11)
- opencv-python>=4.7.0
- numpy>=1.24.0
- Pillow>=9.0.0

## Installation
```bash
cd "for JooX/GasCylinderCV_yolo11/GasCylinderCV_yolo11"
pip install -r requirements.txt
```

### PyTorch with CUDA (if needed)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Dataset Configuration
The dataset configuration is located at:
`data/dataset/data.yaml`

## Model Outputs
- Trained models are saved in `runs/train/`
- Detection results are saved in `runs/detect/`

## Notes for Claude
- Always use the full path when referencing files due to spaces in directory names
- Use quotes around paths in commands: `cd "for JooX/GasCylinderCV_yolo11/GasCylinderCV_yolo11"`
- Check PROJECT_GUIDE.md for detailed implementation information
- The project uses YOLOv11 for object detection
- Multiple confidence threshold tests have been run (0.1, 0.3, 0.5, 0.7, 0.9)