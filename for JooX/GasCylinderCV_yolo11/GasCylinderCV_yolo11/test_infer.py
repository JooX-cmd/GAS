from ultralytics import YOLO
import argparse, os
p = argparse.ArgumentParser(); p.add_argument('--img', required=True)
args = p.parse_args()
weights = r"runs\\train\\cylinder_detector\\weights\\best.pt"
if not os.path.exists(weights):
    weights = "yolo11n.pt"
print("Using weights:", weights)
print("Predicting on:", args.img)
model = YOLO(weights)
res = model.predict(source=args.img, conf=0.6, device='cpu', save=True, verbose=True)
print("Saved to:", res[0].save_dir)
