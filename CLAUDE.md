# CLAUDE.md - Gas Cylinder Detection System

## Project Overview
This is a YOLOv11-based gas cylinder detection system for computer vision applications.

## Project Structure
```
GasCylinderCV_yolo11/
├── data/dataset/          # Training and test datasets
├── runs/                  # Training and detection results
├── src/                   # Source code
│   ├── train_model.py     # Model training script
│   ├── test_model.py      # Model testing script
│   ├── complete_system.py # Complete detection system
│   └── ultra_strict_detector.py # Strict detection variant
└── PROJECT_GUIDE.md       # Detailed project documentation
```

## Key Commands

### Training
```bash
cd "for JooX/GasCylinderCV_yolo11/GasCylinderCV_yolo11"
python src/train_model.py
```

### Testing
```bash
cd "for JooX/GasCylinderCV_yolo11/GasCylinderCV_yolo11"
python src/test_model.py
```

### Complete System
```bash
cd "for JooX/GasCylinderCV_yolo11/GasCylinderCV_yolo11"
python src/complete_system.py
```

## Dependencies
- ultralytics (YOLOv11)
- opencv-python
- torch
- torchvision
- pillow
- numpy

## Installation
```bash
pip install ultralytics opencv-python torch torchvision pillow numpy
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