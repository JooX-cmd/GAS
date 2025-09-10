#!/usr/bin/env python3
"""
Complete Gas Cylinder Detection System
====================================

Integrated system combining training, testing, and real-time detection
for gas cylinder identification using YOLO11.

Author: Gas Cylinder CV Team
Version: 2.0
"""

import argparse
import sys
import logging
import time
from pathlib import Path
import torch
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GasCylinderSystem:
    """Complete gas cylinder detection system."""
    
    def __init__(self, weights_path="runs/train/cylinder_detector/weights/best.pt"):
        self.weights_path = weights_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"System initialized with device: {self.device}")
    
    def load_model(self):
        """Load the trained model."""
        if not Path(self.weights_path).exists():
            logger.error(f"Model weights not found: {self.weights_path}")
            logger.info("Please train a model first using: python src/train_model.py")
            return False
        
        try:
            self.model = YOLO(self.weights_path)
            logger.info(f"Model loaded successfully from: {self.weights_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_image(self, image_path, conf=0.5, save_results=True):
        """Predict gas cylinders in a single image."""
        if not self.model:
            if not self.load_model():
                return None
        
        if not Path(image_path).exists():
            logger.error(f"Image not found: {image_path}")
            return None
        
        try:
            results = self.model.predict(
                source=image_path,
                conf=conf,
                device=self.device,
                save=save_results,
                verbose=False
            )
            
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            logger.info(f"Found {detections} gas cylinders in {image_path}")
            
            return results[0]
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def run_webcam_detection(self, conf=0.5, source=0):
        """Run real-time detection on webcam."""
        if not self.model:
            if not self.load_model():
                return
        
        import cv2
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Could not open camera {source}")
            return
        
        logger.info("Starting webcam detection. Press 'q' to quit, 's' to save frame")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Run detection
                results = self.model.predict(
                    source=frame,
                    conf=conf,
                    device=self.device,
                    verbose=False
                )
                
                # Draw detections
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"Gas Cylinder: {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Gas Cylinder Detection", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Frame saved as {filename}")
        
        except KeyboardInterrupt:
            logger.info("Detection stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def train_new_model(self, data_yaml, epochs=20):
        """Train a new model with the given dataset."""
        from train_model import train_model
        
        try:
            results = train_model(data_yaml, epochs)
            logger.info("Training completed successfully!")
            return results
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    def evaluate_model(self, data_yaml):
        """Evaluate the model on validation dataset."""
        if not self.model:
            if not self.load_model():
                return None
        
        try:
            results = self.model.val(data=data_yaml, device=self.device)
            logger.info("Model evaluation completed")
            return results
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Complete Gas Cylinder Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on webcam
  python complete_system.py --mode webcam
  
  # Test on image
  python complete_system.py --mode image --source test.jpg
  
  # Train new model
  python complete_system.py --mode train --data data/dataset/data.yaml --epochs 50
  
  # Evaluate model
  python complete_system.py --mode eval --data data/dataset/data.yaml
        """
    )
    
    parser.add_argument("--mode", type=str, required=True,
                       choices=["webcam", "image", "train", "eval"],
                       help="Operation mode")
    parser.add_argument("--source", type=str, default="0",
                       help="Image file path or camera index for webcam mode")
    parser.add_argument("--weights", type=str, 
                       default="runs/train/cylinder_detector/weights/best.pt",
                       help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("--data", type=str, default="data/dataset/data.yaml",
                       help="Dataset YAML file path")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Initialize system
    system = GasCylinderSystem(weights_path=args.weights)
    
    try:
        if args.mode == "webcam":
            source = int(args.source) if args.source.isdigit() else args.source
            system.run_webcam_detection(conf=args.conf, source=source)
            
        elif args.mode == "image":
            if args.source == "0":
                logger.error("Please specify image path with --source")
                sys.exit(1)
            result = system.predict_image(args.source, conf=args.conf)
            if result is None:
                sys.exit(1)
                
        elif args.mode == "train":
            result = system.train_new_model(args.data, args.epochs)
            if result is None:
                sys.exit(1)
                
        elif args.mode == "eval":
            result = system.evaluate_model(args.data)
            if result is None:
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
