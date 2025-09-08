import argparse
from ultralytics import YOLO
from pathlib import Path

def train_model(data_yaml, epochs=50, imgsz=640, batch=16):
    print(f"[train] Starting training with {epochs} epochs")
    print(f"[train] Data: {data_yaml}")
    print(f"[train] Image size: {imgsz}, Batch size: {batch}")
    
    # Load model
    model = YOLO("yolo11n.pt")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",  # Use CPU for training
        project="runs/train",
        name="cylinder_detector",
        exist_ok=True
    )
    
    print(f"[train] Training completed!")
    print(f"[train] Results saved to: runs/train/cylinder_detector/")
    
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/dataset/data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    
    if not Path(args.data).exists():
        print(f"Error: Data file {args.data} not found")
        return
    
    train_model(args.data, args.epochs, args.imgsz, args.batch)

if __name__ == "__main__":
    main()
