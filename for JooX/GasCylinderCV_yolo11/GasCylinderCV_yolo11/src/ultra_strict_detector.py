#!/usr/bin/env python3
"""Ultra-Strict Gas Cylinder Detector - Simple Detection"""

import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO

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

class GasCylinderDetector:
    def __init__(self, weights, conf=0.75):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        weights_path = find_model_weights(weights)
        self.model = YOLO(str(weights_path))
        self.conf = conf
        
        print(f"✅ Model: {weights_path}")
        print(f"✅ Device: {'CUDA' if self.device == 0 else 'CPU'}")
        print(f"🚫 STRICT MODE - Only real gas cylinders!")

    def is_phone_or_handheld(self, x1, y1, x2, y2, frame):
        """Detect phones/handheld objects."""
        w, h = x2 - x1, y2 - y1
        area = w * h
        aspect_ratio = h / (w + 1e-6)
        
        # Quick size/ratio checks
        if area < 20000 or w < 90 or 1.5 <= aspect_ratio <= 2.5:
            return True
            
        # Edge detection for rectangular shapes (phones)
        try:
            roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if roi.size == 0:
                return False
                
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4 and cv2.contourArea(contour) > roi.shape[0] * roi.shape[1] * 0.3:
                    return True
        except:
            pass
        return False

    def is_human_part(self, x1, y1, x2, y2, frame):
        """Detect human hands/arms."""
        w, h = x2 - x1, y2 - y1
        aspect_ratio = h / (w + 1e-6)
        
        # Quick checks for human-like proportions
        if w * h < 15000 or aspect_ratio < 1.5:
            return True
            
        # Position analysis - hands usually at edges
        frame_h, frame_w = frame.shape[:2]
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        edge_distance = min(center_x / frame_w, (frame_w - center_x) / frame_w,
                           center_y / frame_h, (frame_h - center_y) / frame_h)
        
        if edge_distance < 0.2:  # Close to edges
            return True
            
        # Color analysis for skin-like colors
        try:
            roi = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
            if roi.size == 0:
                return False
                
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Skin color ranges in HSV
            skin_masks = [
                cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255])),
                cv2.inRange(hsv, np.array([160, 20, 70]), np.array([180, 255, 255]))
            ]
            
            skin_ratio = sum(cv2.countNonZero(mask) for mask in skin_masks) / roi.size
            return skin_ratio > 0.3
        except:
            pass
        return False

    def is_gas_cylinder(self, x1, y1, x2, y2, frame):
        """Check if detection looks like real gas cylinder."""
        w, h = x2 - x1, y2 - y1
        aspect_ratio = h / (w + 1e-6)
        
        # Size requirements
        if w < 100 or h < 200 or w > 200:
            return False
            
        # Aspect ratio (cylinders are tall)
        if aspect_ratio < 2.0 or aspect_ratio > 3.0:
            return False
            
        # Position (should be reasonably centered)
        frame_h, frame_w = frame.shape[:2]
        center_x_ratio = ((x1 + x2) / 2) / frame_w
        center_y_ratio = ((y1 + y2) / 2) / frame_h
        
        if not (0.25 <= center_x_ratio <= 0.75 and 0.3 <= center_y_ratio <= 0.7):
            return False
            
        return True

    def detect(self, frame):
        """Detect gas cylinders - simple found/not found."""
        results = self.model(frame, conf=self.conf, verbose=False)
        
        cylinders = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.conf.item() >= self.conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = box.conf.item()
                    
                    # Apply all filters
                    if (not self.is_phone_or_handheld(x1, y1, x2, y2, frame) and
                        not self.is_human_part(x1, y1, x2, y2, frame) and
                        self.is_gas_cylinder(x1, y1, x2, y2, frame)):
                        cylinders.append((x1, y1, x2, y2, conf))
        
        return cylinders

    def run_webcam(self, source=0):
        """Run detection on webcam."""
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {source}")
            return
        
        print("🎥 Press 'q' to quit")
        print("🔍 Real-time gas cylinder detection!")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cylinders = self.detect(frame)
                
                # Draw results
                display_frame = frame.copy()
                
                for x1, y1, x2, y2, conf in cylinders:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(display_frame, f'Gas Cylinder: {conf:.2f}', 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Status - just found or searching
                status = "✅ GAS CYLINDER FOUND!" if cylinders else "🔍 Searching..."
                color = (0, 255, 0) if cylinders else (0, 165, 255)
                cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                cv2.imshow('Gas Cylinder Detector', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Detection stopped")

def main():
    parser = argparse.ArgumentParser(description="Gas cylinder detector")
    parser.add_argument("--weights", default="runs/train/cylinder_detector/weights/best.pt", 
                       help="Model weights path")
    parser.add_argument("--source", default=0, help="Video source (0 for webcam)")
    parser.add_argument("--conf", type=float, default=0.75, help="Confidence threshold")
    
    args = parser.parse_args()
    
    try:
        detector = GasCylinderDetector(args.weights, args.conf)
        detector.run_webcam(args.source)
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()