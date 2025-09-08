import sys
from pathlib import Path
from ultralytics import YOLO
import torch


def main():
    # Configure paths
    project_root = Path(__file__).resolve().parent.parent
    test_images_dir = project_root / 'data' / 'dataset' / 'test' / 'images'
    default_weights = project_root / 'runs' / 'train' / 'cylinder_detector' / 'weights' / 'best.pt'
    fallback_weights = project_root / 'yolo11n.pt'

    # Resolve weights
    weights = None
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if candidate.exists():
            weights = candidate
    if weights is None and default_weights.exists():
        weights = default_weights
    if weights is None and fallback_weights.exists():
        weights = fallback_weights
    if weights is None:
        print('[test10] Error: No weights found. Provide a path or ensure trained weights exist.')
        return 1

    # List first 10 images
    if not test_images_dir.exists():
        print(f"[test10] Error: test images dir not found: {test_images_dir}")
        return 1
    image_paths = sorted([p for p in test_images_dir.glob('*.jpg')])[:10]
    if not image_paths:
        print(f"[test10] Error: No .jpg images in {test_images_dir}")
        return 1

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"[test10] torch={torch.__version__} cuda={torch.cuda.is_available()} device={'cuda:0' if device == 0 else 'cpu'}")
    print(f"[test10] weights={weights}")
    print(f"[test10] testing {len(image_paths)} images from {test_images_dir}")

    model = YOLO(str(weights))

    # Run predictions and save to runs/detect/test_results
    results = model.predict(
        source=[str(p) for p in image_paths],
        conf=0.25,
        device=device,
        verbose=False,
        project=str(project_root / 'runs' / 'detect'),
        name='test_results',
        exist_ok=True,
        save=True,
    )

    # Print per-image summary
    for p, r in zip(image_paths, results):
        num = 0 if r.boxes is None else len(r.boxes)
        print(f"[test10] {p.name}: detections={num}")

    out_dir = project_root / 'runs' / 'detect' / 'test_results'
    print(f"[test10] Annotated images saved to: {out_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


