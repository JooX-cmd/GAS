import cv2
from ultralytics import YOLO
import os

def test_images(model_path, test_folder, output_folder, conf=0.2):
    # Initialize model
    model = YOLO(model_path)
    model.conf = conf
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get test images
    images = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(images)} images to test")
    
    detections = 0
    for img_name in images:
        img_path = os.path.join(test_folder, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        # Run detection
        results = model.predict(source=img, verbose=False)
        result = results[0]
        
        # Draw detections
        if len(result.boxes) > 0:
            detections += 1
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save result
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, img)
        print(f"Processed {img_name} - {'Detected' if len(result.boxes) > 0 else 'No detection'}")
    
    print(f"\nSummary:\nTotal images: {len(images)}\nDetections: {detections}\nDetection rate: {detections/len(images)*100:.1f}%")

if __name__ == "__main__":
    model_path = "../yolo11n.pt"
    test_folder = "../data/dataset/test/images"
    output_folder = "../runs/detect/test_results"
    
    test_images(model_path, test_folder, output_folder, conf=0.2)