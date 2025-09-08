import argparse
import time
   class UltraStrictCylinderDetector:
    def __init__(self, weights, conf_threshold=0.2, k=10, min_ratio=0.5,
                 min_aspect_ratio=1.0, max_aspect_ratio=5.0,
                 min_width=30, max_width=400, min_height=60,
                 min_center_x_ratio=0.05, max_center_x_ratio=0.95,
                 min_center_y_ratio=0.05, max_center_y_ratio=0.95):_init__(self, weights, conf_threshold=0.2, k=10, min_ratio=0.5,
                 min_aspect_ratio=1.0, max_aspect_ratio=5.0,
                 min_width=30, max_width=400, min_height=60,
                 min_center_x_ratio=0.05, max_center_x_ratio=0.95,
                 min_center_y_ratio=0.05, max_center_y_ratio=0.95):pathlib import Path
import cv2
import torch
from ultralytics import YOLO

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

class UltraStrictCylinderDetector:
    def __init__(self, weights, conf_threshold=0.25, k=20, min_ratio=0.5,
                 min_aspect_ratio=1.2, max_aspect_ratio=4.0,
                 min_width=40, max_width=300, min_height=80,
                 min_center_x_ratio=0.1, max_center_x_ratio=0.9,
                 min_center_y_ratio=0.1, max_center_y_ratio=0.9):
        self.device = "cpu"
        self.model = YOLO(weights)
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
        print(f"[ultra_strict] Starting with conf={self.conf_threshold}, k={k}, min_ratio={min_ratio}")
        print(f"[ultra_strict] Ultra-strict mode: Only detects tall, centered objects")

    def detect(self, frame):
        h_frame, w_frame = frame.shape[:2]
        results = self.model.predict(source=frame, conf=self.conf_threshold, device=self.device, verbose=False)
        r = results[0]
        filtered_boxes = []
        if hasattr(r, "boxes") and r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                aspect_ratio = h / (w + 1e-6)
                center_x = (x1 + x2) / 2 / w_frame
                center_y = (y1 + y2) / 2 / h_frame

                # Apply ultra-strict validation
                if (self.min_width <= w <= self.max_width and
                        self.min_height <= h and
                        self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio and
                        self.min_center_x_ratio <= center_x <= self.max_center_x_ratio and
                        self.min_center_y_ratio <= center_y <= self.max_center_y_ratio):
                    filtered_boxes.append(box)

        # Create a new Results object with filtered boxes
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

        has_det = (r.boxes is not None) and len(r.boxes) > 0
        stable = self.validator.update(has_det)
        return r, stable

    def run_webcam(self, source=0, use_directshow=True):
        if use_directshow:
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {source}")
            return

        print("Press 'q' to quit, 's' to save frame, 'c' to change confidence")
        print("Ultra-strict mode: Only tall, centered objects will be detected")
        
        total_detections = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            result, stable = self.detect(frame)
            
            # Draw detections
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw confidence and stability
                    label = f"Cylinder: {conf:.2f}"
                    if stable:
                        label += " [STABLE]"
                        total_detections += 1
                    
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw status
            status = f"Frame: {frame_count} | Stable: {stable} | Total: {total_detections}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Ultra-Strict Gas Cylinder Detector", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"ultra_strict_detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('c'):
                try:
                    new_conf = float(input("Enter new confidence threshold (0.0-1.0): "))
                    if 0.0 <= new_conf <= 1.0:
                        self.conf_threshold = new_conf
                        print(f"Confidence threshold updated to {new_conf}")
                    else:
                        print("Invalid confidence value")
                except ValueError:
                    print("Invalid input")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"[ultra_strict] Session ended. Total detections: {total_detections}")

def test_images(detector, image_folder):
    import os
    from pathlib import Path
    
    images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(images)} images to test")
    
    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Could not read image: {img_path}")
            continue
            
        result, stable = detector.detect(frame)
        
        # Draw detections
        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Cylinder: {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save result
        output_path = os.path.join("runs/detect/test_results", img_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)
        print(f"Processed {img_name} - {'Detected' if result.boxes is not None and len(result.boxes) > 0 else 'No detection'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="../yolo11n.pt")
    ap.add_argument("--source", type=str, default="0")
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--min_ratio", type=float, default=0.9)
    ap.add_argument("--directshow", action="store_true", help="Force DirectShow on Windows webcams")
    ap.add_argument("--test_folder", type=str, help="Path to folder with test images")
    args = ap.parse_args()
    
    src = int(args.source) if args.source.isdigit() else args.source
    detector = UltraStrictCylinderDetector(
        weights=args.weights,
        conf_threshold=args.conf,
        k=args.k,
        min_ratio=args.min_ratio
    )
    
    if args.test_folder:
        test_images(detector, args.test_folder)
    else:
        detector.run_webcam(source=src, use_directshow=args.directshow)

if __name__ == "__main__":
    main()