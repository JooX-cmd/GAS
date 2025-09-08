import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os

class KFramesValidator:
    def __init__(self, k=20, min_ratio=0.9):
        self.k = k
        self.min_ratio = min_ratio
        self.history = []
        self.stable_count = 0

    def update(self, has_detection):
        self.history.append(has_detection)
        if len(self.history) > self.k:
            self.history.pop(0)
        
        if len(self.history) == self.k:
            positive_ratio = sum(self.history) / len(self.history)
            if positive_ratio >= self.min_ratio:
                self.stable_count += 1
                return True
            else:
                self.stable_count = 0
                return False
        return False

def _resolve_weights_path(preferred_weights: str) -> Path:
    """Resolve model weights path by checking several common locations.

    Order:
    1) Exact path as provided
    2) Relative to current working directory
    3) Relative to this script's directory
    4) Project root (one level up from src)
    5) Common defaults: runs/train/.../best.pt, yolo11n.pt
    """
    candidates = []

    # 1-2) As provided, and CWD relative
    candidates.append(Path(preferred_weights))

    # 3) Relative to this script
    script_dir = Path(__file__).resolve().parent
    candidates.append(script_dir / preferred_weights)

    # 4) Project root (one level up from src)
    project_root = script_dir.parent
    candidates.append(project_root / preferred_weights)

    # 5) Common defaults
    candidates.append(project_root / 'runs/train/cylinder_detector/weights/best.pt')
    candidates.append(project_root / 'yolo11n.pt')

    for path in candidates:
        if path.exists():
            return path

    # Nothing found
    return Path()


class UltraStrictCylinderDetector:
    def __init__(self, weights, conf_threshold=0.6, k=20, min_ratio=0.7,
                 min_aspect_ratio=1.8, max_aspect_ratio=3.2,
                 min_width=80, max_width=220, min_height=160,
                 min_center_x_ratio=0.2, max_center_x_ratio=0.8,
                 min_center_y_ratio=0.25, max_center_y_ratio=0.75):
        
        # Auto-detect best device
        self.device = 0 if torch.cuda.is_available() else "cpu"
        
        # Resolve weights path robustly
        weights_path = _resolve_weights_path(weights)
        if not weights_path.exists():
            print(f"[ultra_strict] Error: Could not find model weights. Tried variants of: '{weights}'")
            print("[ultra_strict] Also checked: 'runs/train/cylinder_detector/weights/best.pt' and 'yolo11n.pt' near this script")
            raise FileNotFoundError("No model weights available")
        
        try:
            self.model = YOLO(str(weights_path))
            print(f"[ultra_strict] Model loaded successfully: {weights_path}")
        except Exception as e:
            print(f"[ultra_strict] Error loading model: {e}")
            raise
        
        self.validator = KFramesValidator(k=k, min_ratio=min_ratio)
        self.conf_threshold = conf_threshold
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.min_center_x_ratio = min_center_x_ratio
        self.max_center_x_ratio = max_center_x_ratio
        self.min_center_y_ratio = min_center_y_ratio
        self.max_center_y_ratio = max_center_y_ratio
        
        device_name = "CUDA" if self.device == 0 else "CPU"
        print(f"[ultra_strict] Device: {device_name}")
        print(f"[ultra_strict] 🚫 ANTI-HAND MODE ACTIVATED! 🚫")
        print(f"[ultra_strict] Enhanced filtering: conf≥{self.conf_threshold}, size≥{min_width}x{min_height}")
        print(f"[ultra_strict] Will reject hands, arms, and human body parts")

    def is_likely_human_part(self, x1, y1, x2, y2, frame):
        """Advanced detection for human body parts (hands, arms, etc.)"""
        w, h = x2 - x1, y2 - y1
        
        # Quick size-based filters for typical hand/arm dimensions
        area = w * h
        
        # Typical hand areas in webcam (empirically determined)
        if area < 15000:  # Very small objects (likely hands/fingers)
            return True
            
        # Aspect ratio analysis for hands vs cylinders
        aspect_ratio = h / (w + 1e-6)
        
        # Hands tend to have specific aspect ratios when extended
        if 1.0 <= aspect_ratio <= 2.2:  # Hands are less cylindrical than gas cylinders
            return True
        
        # Extract the detected region for color analysis
        try:
            roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                return False
            
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Comprehensive skin color detection (multiple skin tones)
            skin_masks = []
            
            # Light skin tones
            lower1 = np.array([0, 20, 70], dtype=np.uint8)
            upper1 = np.array([20, 150, 255], dtype=np.uint8)
            skin_masks.append(cv2.inRange(hsv, lower1, upper1))
            
            # Medium skin tones
            lower2 = np.array([0, 25, 80], dtype=np.uint8)
            upper2 = np.array([25, 170, 230], dtype=np.uint8)
            skin_masks.append(cv2.inRange(hsv, lower2, upper2))
            
            # Darker skin tones
            lower3 = np.array([0, 30, 60], dtype=np.uint8)
            upper3 = np.array([25, 150, 200], dtype=np.uint8)
            skin_masks.append(cv2.inRange(hsv, lower3, upper3))
            
            # Additional range for very light skin
            lower4 = np.array([0, 10, 100], dtype=np.uint8)
            upper4 = np.array([15, 100, 255], dtype=np.uint8)
            skin_masks.append(cv2.inRange(hsv, lower4, upper4))
            
            # Combine all skin masks
            combined_mask = np.zeros_like(skin_masks[0])
            for mask in skin_masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            skin_pixels = cv2.countNonZero(combined_mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            skin_ratio = skin_pixels / total_pixels
            
            # If more than 25% looks like skin, probably a human part
            if skin_ratio > 0.25:
                return True
                
        except Exception as e:
            # If color analysis fails, use conservative approach
            pass
        
        return False

    def detect(self, frame):
        """Detect cylinders in frame with ultra-strict validation and anti-hand filtering"""
        h_frame, w_frame = frame.shape[:2]
        
        # Run YOLO detection
        results = self.model.predict(source=frame, conf=self.conf_threshold, device=self.device, verbose=False)
        r = results[0]
        filtered_boxes = []
        
        # Apply geometric filtering
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                aspect_ratio = h / (w + 1e-6)  # Prevent division by zero
                center_x = (x1 + x2) / 2 / w_frame
                center_y = (y1 + y2) / 2 / h_frame
                area = w * h

                # Enhanced validation checks
                size_valid = (self.min_width <= w <= self.max_width and h >= self.min_height)
                aspect_valid = (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio)
                position_valid = (self.min_center_x_ratio <= center_x <= self.max_center_x_ratio and
                                self.min_center_y_ratio <= center_y <= self.max_center_y_ratio)
                
                # NEW: Anti-human detection
                not_human_part = not self.is_likely_human_part(x1, y1, x2, y2, frame)
                
                # Additional area filter (gas cylinders should be substantial)
                sufficient_area = area >= 12800  # Minimum area for a real gas cylinder
                
                # Combine all filters
                if size_valid and aspect_valid and position_valid and not_human_part and sufficient_area:
                    filtered_boxes.append(box)

        # Update results with filtered boxes
        if filtered_boxes and hasattr(r, "boxes") and r.boxes is not None:
            # Find indices of filtered boxes in original results
            box_indices = []
            for filtered_box in filtered_boxes:
                for i, box in enumerate(r.boxes):
                    if torch.equal(box.data, filtered_box.data):
                        box_indices.append(i)
                        break
            
            if box_indices:
                r.boxes = r.boxes[torch.tensor(box_indices)]
            else:
                r.boxes = None
        else:
            r.boxes = None

        # Update temporal validation
        has_det = (r.boxes is not None) and len(r.boxes) > 0
        stable = self.validator.update(has_det)
        
        return r, stable

    def run_webcam(self, source=0, use_directshow=True):
        """Run detection on webcam feed"""
        # Setup camera
        if use_directshow and source == 0:
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {source}")
            print("Try:")
            print("  - Different camera index (--source 1, --source 2)")
            print("  - Remove --directshow flag")
            print("  - Check camera permissions")
            return

        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'c' to change confidence threshold")
        print("  - Press 'h' to toggle hand detection info")
        print("🚫 Anti-Hand Mode: Will reject human body parts")
        print()
        
        total_detections = 0
        frame_count = 0
        show_debug_info = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                    
                frame_count += 1
                result, stable = self.detect(frame)
                
                # Draw detections
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Draw bounding box
                        color = (0, 255, 0) if stable else (0, 255, 255)  # Green if stable, yellow if not
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)  # Thicker line for better visibility
                        
                        # Draw confidence and stability
                        label = f"GAS CYLINDER: {conf:.2f}"
                        if stable:
                            label += " [STABLE]"
                            total_detections += 1
                        
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Draw status information
                status = f"Frame: {frame_count} | Stable: {stable} | Total: {total_detections}"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw confidence threshold
                conf_text = f"Confidence: {self.conf_threshold:.2f} | Anti-Hand: ON"
                cv2.putText(frame, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Debug info (if enabled)
                if show_debug_info:
                    debug_text = f"Min Size: {self.min_width}x{self.min_height} | Aspect: {self.min_aspect_ratio:.1f}-{self.max_aspect_ratio:.1f}"
                    cv2.putText(frame, debug_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow("Ultra-Strict Gas Cylinder Detector (Anti-Hand)", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"gas_cylinder_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('h'):
                    show_debug_info = not show_debug_info
                    print(f"Debug info: {'ON' if show_debug_info else 'OFF'}")
                elif key == ord('c'):
                    print(f"\nCurrent confidence threshold: {self.conf_threshold}")
                    try:
                        new_conf = float(input("Enter new confidence threshold (0.0-1.0): "))
                        if 0.0 <= new_conf <= 1.0:
                            self.conf_threshold = new_conf
                            print(f"Confidence threshold updated to {new_conf}")
                        else:
                            print("Invalid confidence value. Must be between 0.0 and 1.0")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                    except KeyboardInterrupt:
                        break
                        
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"[ultra_strict] Session ended. Total stable detections: {total_detections}")

def main():
    parser = argparse.ArgumentParser(description="Ultra-Strict Gas Cylinder Detector (Anti-Hand)")
    parser.add_argument("--weights", type=str, default="yolo11n.pt",
                        help="Path to model weights")
    parser.add_argument("--source", type=str, default="0", 
                        help="Camera index (0, 1, 2...) or video file path")
    parser.add_argument("--conf", type=float, default=0.6, 
                        help="Confidence threshold (0.0-1.0) - Higher = more strict")
    parser.add_argument("--k", type=int, default=20, 
                        help="Number of frames for stability validation")
    parser.add_argument("--min_ratio", type=float, default=0.7, 
                        help="Minimum ratio for stable detection")
    parser.add_argument("--directshow", action="store_true", 
                        help="Force DirectShow on Windows webcams")
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    # Create detector
    try:
        detector = UltraStrictCylinderDetector(
            weights=args.weights,
            conf_threshold=args.conf,
            k=args.k,
            min_ratio=args.min_ratio
        )
    except Exception as e:
        print(f"Error creating detector: {e}")
        print("Please check that model weights exist or yolo11n.pt is available")
        return
    
    # Run detection
    detector.run_webcam(source=source, use_directshow=args.directshow)

if __name__ == "__main__":
    main()