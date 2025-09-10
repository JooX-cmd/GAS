#!/usr/bin/env python3
"""
Gas Cylinder Detection Model Training Script
============================================

This script trains a YOLO11 model specifically for detecting gas cylinders.
It automatically detects available GPU and optimizes training parameters accordingly.

Author: Gas Cylinder CV Team
Version: 2.0
"""

import argparse
import sys
import logging
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# Try importing torch, fallback if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not found. GPU detection disabled.")
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check system requirements and GPU availability."""
    logger.info("Checking system requirements...")
    
    # Check CUDA availability
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        device = 0
    else:
        logger.warning("No GPU detected. Training will use CPU (slower)")
        device = "cpu"
    
    return device

def validate_data_config(data_yaml_path):
    """Validate the dataset configuration file."""
    data_path = Path(data_yaml_path)
    if not data_path.exists():
        logger.error(f"Dataset configuration file not found: {data_yaml_path}")
        return False
    
    # Check if dataset directories exist
    yaml_dir = data_path.parent
    train_imgs = yaml_dir / "train" / "images"
    train_labels = yaml_dir / "train" / "labels"
    
    if not train_imgs.exists() or not train_labels.exists():
        logger.error("Training dataset directories not found")
        return False
    
    # Count images and labels
    img_count = len(list(train_imgs.glob("*.jpg"))) + len(list(train_imgs.glob("*.png")))
    label_count = len(list(train_labels.glob("*.txt")))
    
    logger.info(f"Found {img_count} images and {label_count} labels")
    
    if img_count < 100:
        logger.warning(f"Only {img_count} images found. Consider adding more data for better performance")
    
    return True

def train_model(data_yaml, epochs=50, imgsz=640, batch=16, patience=50):
    """
    Train a YOLO11 model for gas cylinder detection.
    
    Args:
        data_yaml (str): Path to dataset configuration file
        epochs (int): Number of training epochs
        imgsz (int): Input image size
        batch (int): Batch size
        patience (int): Early stopping patience
    
    Returns:
        Results object from YOLO training
    """
    # Validate inputs
    if not validate_data_config(data_yaml):
        raise ValueError("Invalid dataset configuration")
    
    # Check system requirements
    device = check_system_requirements()
    
    logger.info("=" * 60)
    logger.info("STARTING GAS CYLINDER MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Image size: {imgsz}x{imgsz}")
    logger.info(f"Batch size: {batch}")
    logger.info(f"Device: {'GPU' if device == 0 else 'CPU'}")
    logger.info(f"Early stopping patience: {patience}")
    
    try:
        # Load pre-trained model
        logger.info("Loading YOLO11n base model...")
        model = YOLO("yolo11n.pt")
        
        # Start training
        logger.info("Starting training process...")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project="runs/train",
            name="cylinder_detector",
            exist_ok=True,
            patience=patience,
            save=True,
            save_period=5,  # Save checkpoint every 5 epochs
            cache=True,  # Cache images for faster training
            workers=4,
            verbose=True
        )
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Best weights saved to: runs/train/cylinder_detector/weights/best.pt")
        logger.info(f"Last weights saved to: runs/train/cylinder_detector/weights/last.pt")
        logger.info(f"Training results saved to: runs/train/cylinder_detector/")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        raise

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train YOLO11 model for gas cylinder detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py --data data/dataset/data.yaml --epochs 50
  python train_model.py --epochs 100 --batch 32 --imgsz 800
        """
    )
    
    parser.add_argument("--data", type=str, default="data/dataset/data.yaml",
                       help="Path to dataset YAML configuration file")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Input image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size (default: 16, adjust based on GPU memory)")
    parser.add_argument("--patience", type=int, default=50,
                       help="Early stopping patience (default: 50)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Start training
        train_model(
            data_yaml=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience
        )
        logger.info("Training completed successfully! Check the results in runs/train/cylinder_detector/")
        
    except KeyboardInterrupt:
        logger.warning("Training was interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
