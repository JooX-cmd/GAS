#!/usr/bin/env python3
"""
GPU-Only Gas Cylinder Detection Training - Simplified
====================================================
Optimized YOLOv11 training script for gas cylinder detection.
GPU required - no CPU fallback.
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import os
import time
import json
import torch
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability - required for training."""
    if not torch.cuda.is_available():
        logger.error("❌ CUDA GPU not detected! GPU required for training.")
        logger.error("Solutions:")
        logger.error("  • Install CUDA drivers")
        logger.error("  • Reinstall PyTorch with CUDA support")
        logger.error("  • Use cloud GPU (Colab, AWS, etc.)")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    return gpu_memory

def auto_batch_size(gpu_memory_gb):
    """Calculate optimal batch size based on GPU memory."""
    if gpu_memory_gb >= 12:    return 32  # High-end GPUs
    elif gpu_memory_gb >= 8:   return 24  # Mid-range GPUs  
    elif gpu_memory_gb >= 6:   return 16  # Entry GPUs
    else:                      return 8   # Low memory GPUs

def train_model(data_yaml="data/dataset/data.yaml", epochs=100, batch=None, patience=30, resume=True):
    """
    Train YOLO11 model for gas cylinder detection - GPU only.
    
    Args:
        data_yaml: Path to dataset configuration
        epochs: Number of training epochs
        batch: Batch size (auto-calculated if None)
        patience: Early stopping patience
        resume: Resume from checkpoint if available
    """
    
    # GPU check
    gpu_memory = check_gpu()
    
    # Auto-calculate batch size if not provided
    if batch is None:
        batch = auto_batch_size(gpu_memory)
        logger.info(f"🧠 Auto batch size: {batch} (GPU: {gpu_memory:.1f}GB)")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    logger.info("=" * 50)
    logger.info("🚀 STARTING TRAINING")
    logger.info("=" * 50)
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch: {batch}")
    logger.info(f"Patience: {patience}")
    
    try:
        # Check for checkpoint
        checkpoint = Path("runs/train/cylinder_detector/weights/last.pt")
        if resume and checkpoint.exists():
            logger.info(f"🔄 Resuming from: {checkpoint}")
            model = YOLO(str(checkpoint))
        else:
            logger.info("🎆 Starting fresh with YOLO11n")
            model = YOLO("yolo11n.pt")
        
        # Start training
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            device=0,  # GPU
            project="runs/train",
            name="cylinder_detector",
            exist_ok=True,
            patience=patience,
            save=True,
            save_period=10,
            cache=True,
            workers=4,
            amp=True,  # Mixed precision
            plots=True,
            val=True,
            resume=resume,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01
        )
        
        logger.info("=" * 50)
        logger.info("🎉 TRAINING COMPLETED!")
        logger.info("=" * 50)
        logger.info(f"✅ Best: runs/train/cylinder_detector/weights/best.pt")
        logger.info(f"✅ Last: runs/train/cylinder_detector/weights/last.pt")
        
        # Save summary
        summary = {
            "completed": datetime.now().isoformat(),
            "epochs": epochs,
            "batch_size": batch,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(gpu_memory, 1)
        }
        
        summary_path = Path("runs/train/cylinder_detector/training_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("🎯 Next steps:")
        logger.info("  python src/test_model.py --webcam")
        logger.info("  python src/ultra_strict_detector.py --source 0")
        
        return results
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted - checkpoint saved")
        return None
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"❌ GPU out of memory! Try smaller --batch {batch//2}")
            torch.cuda.empty_cache()
        raise
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.info("💡 Troubleshooting:")
        logger.info("  • Check dataset folders exist")
        logger.info("  • Verify image/label files match") 
        logger.info("  • Try smaller batch size")
        logger.info("  • Check free disk space")
        raise

def main():
    parser = argparse.ArgumentParser(description="GPU-only YOLO11 training for gas cylinders")
    parser.add_argument("--data", default="data/dataset/data.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, help="Batch size (auto if not set)")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore checkpoints)")
    
    args = parser.parse_args()
    
    try:
        print("\n" + "="*50)
        print("  GAS CYLINDER DETECTION TRAINING")
        print("        GPU-Optimized YOLO11")
        print("="*50 + "\n")
        
        train_model(
            data_yaml=args.data,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
            resume=not args.no_resume
        )
        
        print("\n🎉 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()