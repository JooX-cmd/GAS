from ultralytics import YOLO
import numpy as np, cv2
img = (np.random.rand(640, 640, 3) * 255).astype('uint8')
cv2.imwrite('runs/detect/random_input.jpg', img)
model = YOLO('yolo11n.pt')
res = model.predict(source=img, conf=0.25, device='cpu', save=True, verbose=True)
print('Saved to:', res[0].save_dir)
