# شرح كود ultra_strict_detector.py | Code Explanation for ultra_strict_detector.py

## نظرة عامة | Overview
هذا الملف يحتوي على شرح مفصل لكل سطر في كود `ultra_strict_detector.py` باللغتين العربية والإنجليزية - وهو كاشف صارم جداً لأسطوانات الغاز مع فلترة ذكية
This file contains a detailed explanation of every line in the `ultra_strict_detector.py` code in both Arabic and English - an ultra-strict gas cylinder detector with intelligent filtering.

---

## بنية الملف | File Structure

### 1. المعلومات الأساسية والاستيراد | Basic Info and Imports

```python
#!/usr/bin/env python3
```
**العربية:** هذا السطر يخبر النظام أن يستخدم Python 3 لتشغيل هذا الملف  
**English:** This line tells the system to use Python 3 to run this file

```python
"""Ultra-Strict Gas Cylinder Detector - Simple Detection"""
```
**العربية:** وصف مختصر للملف - كاشف صارم جداً لأسطوانات الغاز مع كشف بسيط  
**English:** Brief file description - ultra-strict gas cylinder detector with simple detection

```python
import argparse
```
**العربية:** استيراد مكتبة argparse للتعامل مع المعاملات من سطر الأوامر  
**English:** Import argparse library to handle command-line arguments

```python
from pathlib import Path
```
**العربية:** استيراد Path للتعامل مع مسارات الملفات بطريقة أفضل  
**English:** Import Path for better file path handling

```python
import cv2
```
**العربية:** استيراد OpenCV للتعامل مع معالجة الصور والفيديو  
**English:** Import OpenCV for image and video processing

```python
import torch
```
**العربية:** استيراد PyTorch للتعلم العميق والذكاء الاصطناعي  
**English:** Import PyTorch for deep learning and AI

```python
import numpy as np
```
**العربية:** استيراد NumPy للتعامل مع المصفوفات والعمليات الرياضية  
**English:** Import NumPy for array operations and mathematical computations

```python
from ultralytics import YOLO
```
**العربية:** استيراد نموذج YOLO من مكتبة ultralytics للكشف عن الأجسام  
**English:** Import YOLO model from ultralytics library for object detection

---

### 2. دالة البحث عن النموذج | Model Finding Function

```python
def find_model_weights(weights):
```
**العربية:** تعريف دالة للبحث عن ملف أوزان النموذج في مواقع مختلفة  
**English:** Define function to search for model weights file in different locations

```python
    """Find model weights file."""
```
**العربية:** وصف مختصر للدالة  
**English:** Brief description of the function

```python
    candidates = [
        Path(weights),
        Path(__file__).parent / weights,
        Path(__file__).parent.parent / weights,
        Path(__file__).parent.parent / 'runs/train/cylinder_detector/weights/best.pt',
        Path(__file__).parent.parent / 'yolo11n.pt'
    ]
```
**العربية:** قائمة بالمواقع المحتملة لملف أوزان النموذج (نفس النمط المستخدم في الملفات الأخرى)  
**English:** List of potential locations for model weights file (same pattern used in other files)

```python
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Model weights not found: {weights}")
```
**العربية:** البحث في المواقع المحتملة وإرجاع أول مسار موجود، أو رفع خطأ إذا لم يُوجد  
**English:** Search potential locations and return first existing path, or raise error if not found

---

### 3. فئة كاشف أسطوانات الغاز | Gas Cylinder Detector Class

```python
class GasCylinderDetector:
```
**العربية:** تعريف فئة كاشف أسطوانات الغاز الصارم  
**English:** Define ultra-strict gas cylinder detector class

```python
    def __init__(self, weights, conf=0.75):
```
**العربية:** دالة التهيئة للفئة مع أوزان النموذج ومعدل ثقة عالي (0.75)  
**English:** Class initialization function with model weights and high confidence threshold (0.75)

```python
        self.device = 0 if torch.cuda.is_available() else "cpu"
```
**العربية:** تحديد الجهاز: استخدام GPU إذا كان متاحاً، وإلا استخدام CPU  
**English:** Set device: use GPU if available, otherwise use CPU

```python
        weights_path = find_model_weights(weights)
        self.model = YOLO(str(weights_path))
        self.conf = conf
```
**العربية:** العثور على مسار الأوزان، تحميل النموذج، وحفظ معدل الثقة  
**English:** Find weights path, load model, and store confidence threshold

```python
        print(f"✅ Model: {weights_path}")
        print(f"✅ Device: {'CUDA' if self.device == 0 else 'CPU'}")
        print(f"🚫 STRICT MODE - Only real gas cylinders!")
```
**العربية:** طباعة معلومات النموذج والتأكيد على الوضع الصارم  
**English:** Print model information and confirm strict mode

---

### 4. دالة كشف الهواتف والأجهزة المحمولة | Phone/Handheld Detection Function

```python
    def is_phone_or_handheld(self, x1, y1, x2, y2, frame):
```
**العربية:** تعريف دالة لكشف الهواتف أو الأجهزة المحمولة لتجنب الكشف الخاطئ  
**English:** Define function to detect phones or handheld objects to avoid false positives

```python
        """Detect phones/handheld objects."""
```
**العربية:** وصف مختصر للدالة  
**English:** Brief description of the function

```python
        w, h = x2 - x1, y2 - y1
        area = w * h
        aspect_ratio = h / (w + 1e-6)
```
**العربية:** حساب العرض والارتفاع والمساحة ونسبة العرض إلى الارتفاع  
**English:** Calculate width, height, area, and aspect ratio

```python
        if area < 20000 or w < 90 or 1.5 <= aspect_ratio <= 2.5:
            return True
```
**العربية:** إذا كانت المساحة صغيرة أو العرض قليل أو النسبة تشبه الهاتف، اعتبره هاتف  
**English:** If area is small or width is narrow or ratio resembles phone, consider it a phone

```python
        try:
            roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if roi.size == 0:
                return False
```
**العربية:** استخراج منطقة الاهتمام من الإطار مع التأكد من صحة الحدود  
**English:** Extract region of interest from frame with boundary checking

```python
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
**العربية:** تحويل المنطقة للرمادي، كشف الحواف، والعثور على الكنتورات  
**English:** Convert region to grayscale, detect edges, and find contours

```python
            for contour in contours:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 4 and cv2.contourArea(contour) > roi.shape[0] * roi.shape[1] * 0.3:
                    return True
```
**العربية:** البحث عن أشكال مستطيلة كبيرة (مثل الهواتف) في الكنتورات  
**English:** Search for large rectangular shapes (like phones) in contours

```python
        except:
            pass
        return False
```
**العربية:** في حالة أي خطأ، العودة بـ False (ليس هاتف)  
**English:** In case of any error, return False (not a phone)

---

### 5. دالة كشف أجزاء الجسم البشري | Human Body Parts Detection Function

```python
    def is_human_part(self, x1, y1, x2, y2, frame):
```
**العربية:** تعريف دالة لكشف أجزاء الجسم البشري (خاصة الأيدي والأذرع)  
**English:** Define function to detect human body parts (especially hands and arms)

```python
        """Detect human hands/arms."""
```
**العربية:** وصف مختصر للدالة - كشف الأيدي والأذرع البشرية  
**English:** Brief description of the function - detect human hands/arms

```python
        w, h = x2 - x1, y2 - y1
        aspect_ratio = h / (w + 1e-6)
```
**العربية:** حساب الأبعاد ونسبة العرض إلى الارتفاع  
**English:** Calculate dimensions and aspect ratio

```python
        if w * h < 15000 or aspect_ratio < 1.5:
            return True
```
**العربية:** إذا كانت المساحة صغيرة أو النسبة غير مناسبة للأسطوانة، اعتبره جزء بشري  
**English:** If area is small or ratio inappropriate for cylinder, consider it human part

```python
        frame_h, frame_w = frame.shape[:2]
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        edge_distance = min(center_x / frame_w, (frame_w - center_x) / frame_w,
                           center_y / frame_h, (frame_h - center_y) / frame_h)
        
        if edge_distance < 0.2:  # Close to edges
            return True
```
**العربية:** حساب موقع المركز والمسافة من الحواف - الأيدي عادة قريبة من الحواف  
**English:** Calculate center position and distance from edges - hands are usually near edges

```python
        try:
            roi = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
            if roi.size == 0:
                return False
                
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
```
**العربية:** استخراج منطقة الاهتمام وتحويلها لفضاء HSV لتحليل اللون  
**English:** Extract region of interest and convert to HSV color space for analysis

```python
            skin_masks = [
                cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255])),
                cv2.inRange(hsv, np.array([160, 20, 70]), np.array([180, 255, 255]))
            ]
```
**العربية:** إنشاء أقنعة لكشف ألوان البشرة في فضاء HSV (مجالين للألوان الدافئة)  
**English:** Create masks to detect skin colors in HSV space (two ranges for warm colors)

```python
            skin_ratio = sum(cv2.countNonZero(mask) for mask in skin_masks) / roi.size
            return skin_ratio > 0.3
```
**العربية:** حساب نسبة البكسلات الشبيهة بالبشرة - إذا كانت >30% فهو جزء بشري  
**English:** Calculate ratio of skin-like pixels - if >30% then it's human part

```python
        except:
            pass
        return False
```
**العربية:** في حالة أي خطأ، العودة بـ False  
**English:** In case of any error, return False

---

### 6. دالة التحقق من الأسطوانة الحقيقية | Real Gas Cylinder Verification Function

```python
    def is_gas_cylinder(self, x1, y1, x2, y2, frame):
```
**العربية:** تعريف دالة للتحقق من أن الكشف يشبه أسطوانة غاز حقيقية  
**English:** Define function to verify detection looks like real gas cylinder

```python
        """Check if detection looks like real gas cylinder."""
```
**العربية:** وصف مختصر للدالة  
**English:** Brief description of the function

```python
        w, h = x2 - x1, y2 - y1
        aspect_ratio = h / (w + 1e-6)
```
**العربية:** حساب الأبعاد ونسبة العرض إلى الارتفاع  
**English:** Calculate dimensions and aspect ratio

```python
        if w < 100 or h < 200 or w > 200:
            return False
```
**العربية:** متطلبات الحجم: العرض 100-200 بكسل والارتفاع على الأقل 200 بكسل  
**English:** Size requirements: width 100-200 pixels and height at least 200 pixels

```python
        if aspect_ratio < 2.0 or aspect_ratio > 3.0:
            return False
```
**العربية:** متطلبات النسبة: الأسطوانات طويلة (نسبة 2.0-3.0)  
**English:** Ratio requirements: cylinders are tall (ratio 2.0-3.0)

```python
        frame_h, frame_w = frame.shape[:2]
        center_x_ratio = ((x1 + x2) / 2) / frame_w
        center_y_ratio = ((y1 + y2) / 2) / frame_h
        
        if not (0.25 <= center_x_ratio <= 0.75 and 0.3 <= center_y_ratio <= 0.7):
            return False
```
**العربية:** متطلبات الموقع: يجب أن تكون في وسط الإطار نسبياً (ليس على الحواف)  
**English:** Position requirements: should be relatively centered in frame (not on edges)

```python
        return True
```
**العربية:** إذا مرت جميع الاختبارات، فهي أسطوانة غاز محتملة  
**English:** If all tests pass, it's a potential gas cylinder

---

### 7. دالة الكشف الرئيسية | Main Detection Function

```python
    def detect(self, frame):
```
**العربية:** تعريف دالة الكشف الرئيسية - بسيطة: وُجدت أم لا  
**English:** Define main detection function - simple: found or not found

```python
        """Detect gas cylinders - simple found/not found."""
```
**العربية:** وصف مختصر للدالة  
**English:** Brief description of the function

```python
        results = self.model(frame, conf=self.conf, verbose=False)
```
**العربية:** تشغيل نموذج YOLO على الإطار مع معدل الثقة المحدد  
**English:** Run YOLO model on frame with specified confidence threshold

```python
        cylinders = []
        if results and results[0].boxes is not None:
```
**العربية:** إنشاء قائمة فارغة للأسطوانات والتحقق من وجود نتائج  
**English:** Create empty cylinders list and check if results exist

```python
            for box in results[0].boxes:
                if box.conf.item() >= self.conf:
```
**العربية:** التكرار عبر كل صندوق مُكتشف والتحقق من الثقة  
**English:** Loop through each detected box and check confidence

```python
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = box.conf.item()
```
**العربية:** استخراج إحداثيات الصندوق ومعدل الثقة  
**English:** Extract box coordinates and confidence level

```python
                    if (not self.is_phone_or_handheld(x1, y1, x2, y2, frame) and
                        not self.is_human_part(x1, y1, x2, y2, frame) and
                        self.is_gas_cylinder(x1, y1, x2, y2, frame)):
```
**العربية:** تطبيق جميع الفلاتر:  
- ليس هاتف أو جهاز محمول  
- ليس جزء بشري  
- يشبه أسطوانة غاز  

**English:** Apply all filters:  
- not phone or handheld device  
- not human body part  
- looks like gas cylinder

```python
                        cylinders.append((x1, y1, x2, y2, conf))
```
**العربية:** إذا مرت جميع الفلاتر، إضافة الأسطوانة للقائمة  
**English:** If all filters pass, add cylinder to list

```python
        return cylinders
```
**العربية:** إرجاع قائمة الأسطوانات المُكتشفة والمُفلترة  
**English:** Return list of detected and filtered cylinders

---

### 8. دالة تشغيل الكاميرا | Webcam Function

```python
    def run_webcam(self, source=0):
```
**العربية:** تعريف دالة تشغيل الكشف على الكاميرا المباشرة  
**English:** Define function to run detection on live webcam

```python
        """Run detection on webcam."""
```
**العربية:** وصف مختصر للدالة  
**English:** Brief description of the function

```python
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {source}")
            return
```
**العربية:** فتح الكاميرا والتحقق من نجاح الفتح  
**English:** Open camera and check if opening was successful

```python
        print("🎥 Press 'q' to quit")
        print("🔍 Real-time gas cylinder detection!")
```
**العربية:** طباعة تعليمات للمستخدم  
**English:** Print instructions for user

```python
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
```
**العربية:** حلقة لا نهائية لقراءة الإطارات من الكاميرا  
**English:** Infinite loop to read frames from camera

```python
                cylinders = self.detect(frame)
```
**العربية:** تشغيل الكشف على الإطار الحالي  
**English:** Run detection on current frame

```python
                display_frame = frame.copy()
```
**العربية:** إنشاء نسخة من الإطار للعرض  
**English:** Create copy of frame for display

```python
                for x1, y1, x2, y2, conf in cylinders:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(display_frame, f'Gas Cylinder: {conf:.2f}', 
                              (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
```
**العربية:** رسم مستطيلات خضراء حول الأسطوانات المُكتشفة مع نص يوضح الثقة  
**English:** Draw green rectangles around detected cylinders with confidence text

```python
                status = "✅ GAS CYLINDER FOUND!" if cylinders else "🔍 Searching..."
                color = (0, 255, 0) if cylinders else (0, 165, 255)
                cv2.putText(display_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
```
**العربية:** عرض حالة بسيطة: وُجدت أسطوانة أم يبحث  
**English:** Display simple status: cylinder found or searching

```python
                cv2.imshow('Gas Cylinder Detector', display_frame)
```
**العربية:** عرض الإطار مع النتائج  
**English:** Display frame with results

```python
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
```
**العربية:** الخروج من الحلقة إذا ضُغط على المفتاح 'q'  
**English:** Exit loop if 'q' key is pressed

```python
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("👋 Detection stopped")
```
**العربية:** تنظيف الموارد وإغلاق النوافذ عند الانتهاء  
**English:** Clean up resources and close windows when finished

---

### 9. الدالة الرئيسية | Main Function

```python
def main():
    parser = argparse.ArgumentParser(description="Gas cylinder detector")
```
**العربية:** إنشاء محلل معاملات سطر الأوامر  
**English:** Create command-line argument parser

```python
    parser.add_argument("--weights", default="runs/train/cylinder_detector/weights/best.pt", 
                       help="Model weights path")
    parser.add_argument("--source", default=0, help="Video source (0 for webcam)")
    parser.add_argument("--conf", type=float, default=0.75, help="Confidence threshold")
```
**العربية:** إضافة معاملات سطر الأوامر للأوزان ومصدر الفيديو والثقة  
**English:** Add command-line arguments for weights, video source, and confidence

```python
    args = parser.parse_args()
    
    try:
        detector = GasCylinderDetector(args.weights, args.conf)
        detector.run_webcam(args.source)
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
```
**العربية:** تحليل المعاملات وتشغيل الكاشف مع معالجة الأخطاء  
**English:** Parse arguments and run detector with error handling

```python
if __name__ == "__main__":
    main()
```
**العربية:** تشغيل الدالة الرئيسية إذا تم تشغيل الملف مباشرة  
**English:** Run main function if file is executed directly

---

## خلاصة البرنامج | Program Summary

### الغرض | Purpose
**العربية:** هذا البرنامج مخصص للكشف الصارم جداً عن أسطوانات الغاز في الوقت الفعلي، مع فلترة ذكية لتجنب الإيجابيات الخاطئة من الهواتف وأجزاء الجسم البشري.

**English:** This program is designed for ultra-strict real-time gas cylinder detection, with intelligent filtering to avoid false positives from phones and human body parts.

### الميزات الرئيسية | Key Features
**العربية:**
- كشف في الوقت الفعلي عبر الكاميرا
- فلترة ذكية للهواتف والأجهزة المحمولة
- كشف أجزاء الجسم البشري لتجنب الخطأ
- التحقق من أبعاد ونسب الأسطوانات الحقيقية
- واجهة بسيطة: وُجدت/لم توجد

**English:**
- Real-time detection via camera
- Intelligent filtering for phones and handheld devices
- Human body parts detection to avoid errors
- Verification of real cylinder dimensions and ratios
- Simple interface: found/not found

### كيفية الاستخدام | How to Use
```bash
# كشف أساسي | Basic detection
python src/ultra_strict_detector.py

# مع إعدادات مخصصة | With custom settings
python src/ultra_strict_detector.py --conf 0.8 --source 0

# مع نموذج مخصص | With custom model
python src/ultra_strict_detector.py --weights "custom_model.pt"
```

### المخرجات | Outputs
**العربية:**
- عرض مباشر للكاميرا مع النتائج
- مستطيلات خضراء حول الأسطوانات المُكتشفة
- نص يوضح حالة الكشف ومعدل الثقة
- رسائل واضحة: "وُجدت أسطوانة غاز!" أو "يبحث..."

**English:**
- Live camera display with results
- Green rectangles around detected cylinders
- Text showing detection status and confidence
- Clear messages: "GAS CYLINDER FOUND!" or "Searching..."