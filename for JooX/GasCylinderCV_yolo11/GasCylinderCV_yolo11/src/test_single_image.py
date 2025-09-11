#!/usr/bin/env python3
"""Single Image Gas Cylinder Detector - Simple Testing"""

import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

def find_model_weights(weights):
    """Find model weights file."""
    candidates = [
        Path(weights),
        Path(__file__).parent / weights,
        Path(__file__).parent.parent / weights,
        Path(__file__).parent.parent / 'runs/train/cylinder_detector/weights/best.pt',
        Path(__file__).parent.parent / 'yolo11n.pt'
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Model weights not found: {weights}")

class GasCylinderTester:
    def __init__(self, weights, conf=0.5):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        weights_path = find_model_weights(weights)
        self.model = YOLO(str(weights_path))
        self.conf = conf
        
        print(f"✅ Model: {weights_path}")
        print(f"✅ Device: {'CUDA' if self.device == 0 else 'CPU'}")
        print(f"📊 Confidence: {conf}")
    
    def detect(self, image_path):
        """Detect gas cylinders in image."""
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return []
        
        print(f"🔍 Testing: {Path(image_path).name}")
        
        try:
            results = self.model(image_path, conf=self.conf, verbose=False)
            
            cylinders = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    if box.conf.item() >= self.conf:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = box.conf.item()
                        cylinders.append((x1, y1, x2, y2, conf))
            
            return cylinders
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
    
    def visualize_results(self, image_path, cylinders):
        """Create and save visualization."""
        try:
            # Load image
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create figure
            plt.figure(figsize=(12, 8))
            plt.imshow(img_rgb)
            
            # Draw detections
            for i, (x1, y1, x2, y2, conf) in enumerate(cylinders):
                color = 'lime' if conf >= 0.7 else 'yellow' if conf >= 0.5 else 'orange'
                
                # Draw bounding box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color=color, linewidth=3)
                plt.gca().add_patch(rect)
                
                # Add label
                plt.text(x1, y1-10, f'Cylinder {i+1}: {conf:.1%}', 
                        fontsize=12, color=color, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            plt.title(f'Gas Cylinder Detection\n{Path(image_path).name}', 
                     fontsize=16, weight='bold')
            plt.axis('off')
            
            # Save result
            output_name = f"detection_{Path(image_path).stem}.png"
            plt.savefig(output_name, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            print(f"💾 Saved: {output_name}")
            plt.show()
            
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
    
    def test_image(self, image_path):
        """Test single image and show results."""
        cylinders = self.detect(image_path)
        
        if cylinders:
            print(f"✅ Found {len(cylinders)} gas cylinder(s)!")
            for i, (x1, y1, x2, y2, conf) in enumerate(cylinders):
                print(f"  Cylinder {i+1}: {conf:.1%} confidence at ({x1},{y1})-({x2},{y2})")
            self.visualize_results(image_path, cylinders)
        else:
            print("❌ No gas cylinders detected")
        
        return cylinders
def main():
    parser = argparse.ArgumentParser(description="Gas cylinder single image tester")
    parser.add_argument("--weights", default="runs/train/cylinder_detector/weights/best.pt", 
                       help="Model weights path")
    parser.add_argument("--image", required=True, help="Image path to test")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    try:
        tester = GasCylinderTester(args.weights, args.conf)
        tester.test_image(args.image)
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()