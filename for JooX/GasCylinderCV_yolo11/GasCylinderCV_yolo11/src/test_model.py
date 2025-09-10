import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import torch

def test_model(weights, image_path=None, webcam=False, conf=0.5):
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[env] torch={torch.__version__}  cuda? {torch.cuda.is_available()}  device={device}")
    print(f"[load] weights: {weights}")
    
    model = YOLO(weights)
    
    if image_path and Path(image_path).exists():
        # Test on image
        results = model.predict(source=image_path, conf=conf, device=device, verbose=False)
        r = results[0]
        
        print(f"[result] detections={len(r.boxes) if r.boxes is not None else 0}")
        
        # Save annotated image
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                conf_score = float(box.conf[0])
                print(f"  Detection: confidence={conf_score:.3f}")
        
        # Save result
        r.save()
        print(f"[saved] annotated => {r.save_dir}")
        
    elif webcam:
        # Test on webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        print("Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model.predict(source=frame, conf=conf, device=device, verbose=False)
            r = results[0]
            
            # Draw detections
            if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf_score = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Cylinder: {conf_score:.2f}", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Model Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Error: No valid image path or webcam specified")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default="runs/train/cylinder_detector/weights/best.pt")
    ap.add_argument("--image", type=str, help="Path to test image")
    ap.add_argument("--webcam", action="store_true", help="Test on webcam")
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = ap.parse_args()
    
    test_model(args.weights, args.image, args.webcam, args.conf)

if __name__ == "__main__":
    main()
