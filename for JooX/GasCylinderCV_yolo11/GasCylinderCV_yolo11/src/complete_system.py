import argparse
import time
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

class KFramesValidator:
    def __init__(self, k=15, min_ratio=0.8):
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

def run(weights, source=0, conf=0.7, k=15, min_ratio=0.8, use_directshow=True):
    device = 0 if torch.cuda.is_available() else "cpu"

    # Ensure weights exist, fallback to base model
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"[complete_system] Warning: weights not found: {weights_path.resolve() if weights_path.parent.exists() else weights_path}")
        print("[complete_system] Falling back to 'yolo11n.pt'")
        weights_path = Path("yolo11n.pt")

    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        print(f"[complete_system] Failed to load model: {e}")
        return
    validator = KFramesValidator(k=k, min_ratio=min_ratio)
    
    print(f"[complete_system] device = {'cuda:0' if device == 0 else 'cpu'}")
    print("Press 'q' to quit, 's' to save frame.")
    
    if use_directshow:
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {source}")
        return

    total_detections = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        results = model.predict(source=frame, conf=conf, device=device, verbose=False)
        r = results[0]
        
        has_det = (r.boxes is not None) and len(r.boxes) > 0
        stable = validator.update(has_det)
        
        # Draw detections
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_score = float(box.conf[0])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence and stability
                label = f"Cylinder: {conf_score:.2f}"
                if stable:
                    label += " [STABLE]"
                    total_detections += 1
                
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw status
        status = f"Frame: {frame_count} | Stable: {stable} | Total: {total_detections}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Gas Cylinder Detector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"detection_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"[complete_system] Session ended. Total detections: {total_detections}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="runs/train/cylinder_detector/weights/best.pt")
    ap.add_argument("--source", type=str, default="0")  # int index or path/URL
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--min_ratio", type=float, default=0.8)
    ap.add_argument("--directshow", action="store_true", help="Force DirectShow on Windows webcams")
    args = ap.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    run(args.weights, source=src, conf=args.conf, k=args.k, min_ratio=args.min_ratio, use_directshow=args.directshow)
