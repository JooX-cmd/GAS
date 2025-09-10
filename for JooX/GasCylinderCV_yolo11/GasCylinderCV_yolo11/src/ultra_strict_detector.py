import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os

class KFramesValidator:
    def __init__(self, k=25, min_ratio=0.8):
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
    """Resolve model weights path by checking several common locations."""
    candidates = []

    # As provided, and CWD relative
    candidates.append(Path(preferred_weights))

    # Relative to this script
    script_dir = Path(__file__).resolve().parent
    candidates.append(script_dir / preferred_weights)

    # Project root (one level up from src)
    project_root = script_dir.parent
    candidates.append(project_root / preferred_weights)

    # Common defaults
    candidates.append(project_root / 'runs/train/cylinder_detector/weights/best.pt')
    candidates.append(project_root / 'yolo11n.pt')

    for path in candidates:
        if path.exists():
            return path

    return Path()

class SuperStrictCylinderDetector:
    def __init__(self, weights, conf_threshold=0.75, k=25, min_ratio=0.8,
                 min_aspect_ratio=2.0, max_aspect_ratio=3.0,
                 min_width=100, max_width=200, min_height=200,
                 min_center_x_ratio=0.25, max_center_x_ratio=0.75,
                 min_center_y_ratio=0.3, max_center_y_ratio=0.7):
        
        # Auto-detect best device
        self.device = 0 if torch.cuda.is_available() else "cpu"
        
        # Resolve weights path robustly
        weights_path = _resolve_weights_path(weights)
        if not weights_path.exists():
            print(f"[super_strict] Error: Could not find model weights. Tried variants of: '{weights}'")
            print("[super_strict] Also checked: 'runs/train/cylinder_detector/weights/best.pt' and 'yolo11n.pt' near this script")
            raise FileNotFoundError("No model weights available")
        
        try:
            self.model = YOLO(str(weights_path))
            print(f"[super_strict] Model loaded successfully: {weights_path}")
        except Exception as e:
            print(f"[super_strict] Error loading model: {e}")
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
        print(f"[super_strict] Device: {device_name}")
        print(f"[super_strict] 🚫🚫 SUPER-STRICT MODE ACTIVATED! 🚫🚫")
        print(f"[super_strict] Will reject: hands, phones, bottles, and small objects")
        print(f"[super_strict] ONLY REAL GAS CYLINDERS: conf≥{self.conf_threshold}, size≥{min_width}x{min_height}")

    def is_likely_phone_or_handheld(self, x1, y1, x2, y2, frame):
        """Detect phones, remotes, and other handheld rectangular objects"""
        w, h = x2 - x1, y2 - y1
        area = w * h
        aspect_ratio = h / (w + 1e-6)
        
        # Phone/rectangle detection criteria
        
        # 1. Size analysis - phones are typically smaller than gas cylinders
        if area < 20000:  # Small objects (phones, remotes, etc.)
            return True
            
        # 2. Aspect ratio analysis - phones have specific ratios
        if 1.5 <= aspect_ratio <= 2.5:  # Typical phone aspect ratios
            return True
            
        # 3. Width analysis - phones are narrower than gas cylinders
        if w < 90:  # Very narrow objects (likely phones)
            return True
            
        # 4. Extract region for detailed analysis
        try:
            roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if roi.size == 0 or roi.shape[0] < 20 or roi.shape[1] < 20:
                return False
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Edge detection to find rectangular shapes (typical of phones/screens)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular contours (phones have very rectangular shapes)
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If we find a 4-sided polygon (rectangle), likely a phone
                if len(approx) == 4:
                    contour_area = cv2.contourArea(contour)
                    roi_area = roi.shape[0] * roi.shape[1]
                    
                    # If the rectangular contour covers significant portion, it's likely a phone
                    if contour_area > roi_area * 0.3:
                        return True
            
            # Color analysis for phone detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Check for very dark colors (common in phone cases/screens)
            dark_threshold = 50
            dark_pixels = np.sum(hsv[:,:,2] < dark_threshold)
            total_pixels = roi.shape[0] * roi.shape[1]
            dark_ratio = dark_pixels / total_pixels
            
            # If very dark (like phone screen), likely not a gas cylinder
            if dark_ratio > 0.7:
                return True
                
        except Exception as e:
            pass
        
        return False

    def is_likely_human_part(self, x1, y1, x2, y2, frame):
        """Advanced detection for human body parts (hands, arms, etc.)"""
        w, h = x2 - x1, y2 - y1
        area = w * h
        
        # Quick size-based filters for typical hand/arm dimensions
        if area < 18000:  # Very small objects (likely hands/fingers)
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
            
            # Multiple skin tone ranges
            skin_ranges = [
                ([0, 20, 70], [20, 150, 255]),    # Light skin
                ([0, 25, 80], [25, 170, 230]),    # Medium skin
                ([0, 30, 60], [25, 150, 200]),    # Darker skin
                ([0, 10, 100], [15, 100, 255])   # Very light skin
            ]
            
            skin_pixels = 0
            total_pixels = roi.shape[0] * roi.shape[1]
            
            for lower, upper in skin_ranges:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
                skin_pixels += cv2.countNonZero(mask)
            
            skin_ratio = skin_pixels / total_pixels
            
            # If more than 20% looks like skin, probably a human part
            if skin_ratio > 0.20:
                return True
                
        except Exception as e:
            pass
        
        return False

    def is_likely_gas_cylinder(self, x1, y1, x2, y2, frame):
        """Positive identification of gas cylinder characteristics"""
        w, h = x2 - x1, y2 - y1
        area = w * h
        aspect_ratio = h / (w + 1e-6)
        
        # Gas cylinder must meet ALL these criteria:
        
        # 1. Substantial size (real gas cylinders are big)
        if area < 25000:
            return False
            
        # 2. Very specific aspect ratio (gas cylinders are very tall and narrow)
        if not (2.2 <= aspect_ratio <= 2.8):
            return False
            
        # 3. Minimum dimensions
        if w < 110 or h < 220:
            return False
            
        # 4. Maximum dimensions (not too big to be believable)
        if w > 180 or h > 400:
            return False
        
        try:
            # Extract region for advanced analysis
            roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if roi.size == 0:
                return False
            
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Gas cylinders typically have:
            # - Metallic colors (gray, silver, blue, green)
            # - Consistent color throughout
            # - Not skin-colored
            # - Not very dark (like phone screens)
            
            # Check for metallic/industrial colors
            # Avoid skin tones and very dark colors
            
            # Calculate color consistency (gas cylinders have uniform color)
            h_channel = hsv[:,:,0]
            s_channel = hsv[:,:,1]
            v_channel = hsv[:,:,2]
            
            h_std = np.std(h_channel)
            s_std = np.std(s_channel)
            v_std = np.std(v_channel)
            
            # Gas cylinders have more consistent color than complex objects
            color_consistency = (h_std < 30 and s_std < 50 and v_std < 60)
            
            # Avoid very dark objects (phones/screens)
            avg_brightness = np.mean(v_channel)
            not_too_dark = avg_brightness > 60
            
            # Avoid skin colors
            skin_pixels = 0
            total_pixels = roi.shape[0] * roi.shape[1]
            
            # Check against skin color ranges
            skin_ranges = [
                ([0, 20, 70], [20, 150, 255]),
                ([0, 25, 80], [25, 170, 230]),
                ([0, 30, 60], [25, 150, 200])
            ]
            
            for lower, upper in skin_ranges:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
                skin_pixels += cv2.countNonZero(mask)
            
            skin_ratio = skin_pixels / total_pixels
            not_skin_colored = skin_ratio < 0.15
            
            return color_consistency and not_too_dark and not_skin_colored
            
        except Exception as e:
            return False

    def detect(self, frame):
        """Detect cylinders with super-strict validation"""
        h_frame, w_frame = frame.shape[:2]
        
        # Run YOLO detection
        results = self.model.predict(source=frame, conf=self.conf_threshold, device=self.device, verbose=False)
        r = results[0]
        filtered_boxes = []
        
        # Apply super-strict filtering
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                aspect_ratio = h / (w + 1e-6)
                center_x = (x1 + x2) / 2 / w_frame
                center_y = (y1 + y2) / 2 / h_frame
                area = w * h

                # Basic geometric validation
                size_valid = (self.min_width <= w <= self.max_width and h >= self.min_height)
                aspect_valid = (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio)
                position_valid = (self.min_center_x_ratio <= center_x <= self.max_center_x_ratio and
                                self.min_center_y_ratio <= center_y <= self.max_center_y_ratio)
                
                # Advanced filtering - REJECT these:
                is_phone = self.is_likely_phone_or_handheld(x1, y1, x2, y2, frame)
                is_human = self.is_likely_human_part(x1, y1, x2, y2, frame)
                
                # Positive validation - MUST BE gas cylinder
                is_cylinder = self.is_likely_gas_cylinder(x1, y1, x2, y2, frame)
                
                # Only accept if it passes ALL tests
                if (size_valid and aspect_valid and position_valid and 
                    not is_phone and not is_human and is_cylinder):
                    filtered_boxes.append(box)

        # Update results with filtered boxes
        if filtered_boxes and hasattr(r, "boxes") and r.boxes is not None:
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
        if use_directshow and source == 0:
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {source}")
            return

        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'c' to change confidence threshold")
        print("🚫🚫 SUPER-STRICT MODE: Only REAL gas cylinders detected")
        print("Will reject: hands, phones, bottles, remotes, and small objects")
        print()
        
        total_detections = 0
        frame_count = 0
        
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
                        
                        # Draw bounding box - very prominent for real detections
                        color = (0, 255, 0) if stable else (0, 255, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                        
                        # Draw confidence and stability
                        label = f"REAL GAS CYLINDER: {conf:.2f}"
                        if stable:
                            label += " [CONFIRMED]"
                            total_detections += 1
                        
                        cv2.putText(frame, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Draw status information
                status = f"Frame: {frame_count} | Confirmed: {stable} | Total Real Cylinders: {total_detections}"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw mode information
                mode_text = f"SUPER-STRICT MODE | Conf: {self.conf_threshold:.2f} | Anti-Phone: ON"
                cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Super-Strict Gas Cylinder Detector (No Phones/Hands)", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"real_gas_cylinder_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as {filename}")
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
            print(f"[super_strict] Session ended. Total REAL gas cylinders detected: {total_detections}")

def main():
    parser = argparse.ArgumentParser(description="Super-Strict Gas Cylinder Detector (No Phones/Hands)")
    parser.add_argument("--weights", type=str, default="yolo11n.pt",
                        help="Path to model weights")
    parser.add_argument("--source", type=str, default="0", 
                        help="Camera index (0, 1, 2...) or video file path")
    parser.add_argument("--conf", type=float, default=0.85, 
                        help="Confidence threshold (0.0-1.0) - Very high for super-strict mode")
    parser.add_argument("--k", type=int, default=25, 
                        help="Number of frames for stability validation")
    parser.add_argument("--min_ratio", type=float, default=0.8, 
                        help="Minimum ratio for stable detection")
    parser.add_argument("--directshow", action="store_true", 
                        help="Force DirectShow on Windows webcams")
    
    args = parser.parse_args()
    
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    try:
        detector = SuperStrictCylinderDetector(
            weights=args.weights,
            conf_threshold=args.conf,
            k=args.k,
            min_ratio=args.min_ratio
        )
    except Exception as e:
        print(f"Error creating detector: {e}")
        return
    
    detector.run_webcam(source=source, use_directshow=args.directshow)

if __name__ == "__main__":
    main()