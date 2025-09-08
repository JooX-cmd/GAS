import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def main():
    parser = argparse.ArgumentParser(description='Predict on images and save annotations')
    parser.add_argument('--weights', type=str, help='Path to model weights (defaults to trained best.pt)')
    parser.add_argument('--dir', type=str, default='test', choices=['test', 'train', 'val'], help='Dataset split to use')
    parser.add_argument('--limit', type=int, default=10, help='Number of images to process (0 = all)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--batch', type=int, default=1, help='Batch size for prediction')
    args = parser.parse_args()

    # Configure paths
    project_root = Path(__file__).resolve().parent.parent
    images_dir = project_root / 'data' / 'dataset' / args.dir / 'images'
    default_weights = project_root / 'runs' / 'train' / 'cylinder_detector' / 'weights' / 'best.pt'
    fallback_weights = project_root / 'yolo11n.pt'

    # Resolve weights
    weights = None
    if args.weights:
        candidate = Path(args.weights)
        if candidate.exists():
            weights = candidate
    if weights is None and default_weights.exists():
        weights = default_weights
    if weights is None and fallback_weights.exists():
        weights = fallback_weights
    if weights is None:
        print('[test] Error: No weights found. Provide --weights or ensure trained weights exist.')
        return 1

    # List images with limit
    if not images_dir.exists():
        print(f"[test] Error: images dir not found: {images_dir}")
        return 1
    all_images = sorted([p for p in images_dir.glob('*.jpg')])
    if not all_images:
        print(f"[test] Error: No .jpg images in {images_dir}")
        return 1
    image_paths = all_images if args.limit == 0 else all_images[: args.limit]

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"[test] torch={torch.__version__} cuda={torch.cuda.is_available()} device={'cuda:0' if device == 0 else 'cpu'}")
    print(f"[test] weights={weights}")
    print(f"[test] split={args.dir} total={len(image_paths)} (limit={args.limit}) from {images_dir}")

    model = YOLO(str(weights))

    # Stream predictions to avoid high RAM usage
    source = str(images_dir) if args.limit == 0 else [str(p) for p in image_paths]
    results_iter = model.predict(
        source=source,
        conf=args.conf,
        device=device,
        verbose=False,
        project=str(project_root / 'runs' / 'detect'),
        name='test_results',
        exist_ok=True,
        save=True,
        stream=True,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
    )

    # Print per-image summary as they stream
    for r in results_iter:
        num = 0 if r.boxes is None else len(r.boxes)
        # r.path is full path string to the image
        print(f"[test] {Path(r.path).name}: detections={num}")

    out_dir = project_root / 'runs' / 'detect' / 'test_results'
    print(f"[test] Annotated images saved to: {out_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


