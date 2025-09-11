# شرح كود test_single_image.py | Code Explanation for test_single_image.py

## نظرة عامة | Overview
هذا الملف يحتوي على شرح مفصل لكل سطر في كود `test_single_image.py` باللغتين العربية والإنجليزية
This file contains a detailed explanation of every line in the `test_single_image.py` code in both Arabic and English.

---

## بنية الملف | File Structure

### 1. الاستيراد والمكتبات | Imports and Libraries

```python
#!/usr/bin/env python3
```
**العربية:** هذا السطر يخبر النظام أن يستخدم Python 3 لتشغيل هذا الملف  
**English:** This line tells the system to use Python 3 to run this file

```python
"""Single Image Gas Cylinder Detector - Simple Testing"""
```
**العربية:** وصف مختصر للملف - كاشف أسطوانات الغاز للصور المفردة  
**English:** Brief description of the file - Gas Cylinder Detector for single images

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
**العربية:** استيراد OpenCV للتعامل مع معالجة الصور  
**English:** Import OpenCV for image processing

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

```python
import matplotlib.pyplot as plt
```
**العربية:** استيراد matplotlib لرسم وعرض النتائج البصرية  
**English:** Import matplotlib for plotting and visualizing results

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
**العربية:** قائمة بالمواقع المحتملة لملف أوزان النموذج:  
- المسار المُدخل مباشرة  
- في نفس مجلد الملف الحالي  
- في المجلد الأب  
- في مجلد النتائج المُدرب  
- ملف النموذج الأساسي yolo11n.pt  

**English:** List of potential locations for model weights file:  
- Direct input path  
- Same folder as current file  
- Parent folder  
- Trained results folder  
- Base model file yolo11n.pt

```python
    for path in candidates:
        if path.exists():
            return path
```
**العربية:** التكرار عبر المواقع المحتملة وإرجاع أول مسار موجود  
**English:** Loop through potential locations and return first existing path

```python
    raise FileNotFoundError(f"Model weights not found: {weights}")
```
**العربية:** رفع خطأ إذا لم يتم العثور على ملف الأوزان في أي موقع  
**English:** Raise error if weights file not found in any location

---

### 3. فئة كاشف أسطوانات الغاز | Gas Cylinder Tester Class

```python
class GasCylinderTester:
```
**العربية:** تعريف فئة لاختبار كشف أسطوانات الغاز  
**English:** Define class for testing gas cylinder detection

```python
    def __init__(self, weights, conf=0.5):
```
**العربية:** دالة التهيئة للفئة مع معاملات الأوزان ومعدل الثقة  
**English:** Class initialization function with weights and confidence parameters

```python
        self.device = 0 if torch.cuda.is_available() else "cpu"
```
**العربية:** تحديد الجهاز: استخدام GPU إذا كان متاحاً، وإلا استخدام CPU  
**English:** Set device: use GPU if available, otherwise use CPU

```python
        weights_path = find_model_weights(weights)
```
**العربية:** البحث عن مسار ملف أوزان النموذج باستخدام الدالة المُعرّفة سابقاً  
**English:** Find model weights path using previously defined function

```python
        self.model = YOLO(str(weights_path))
```
**العربية:** تحميل نموذج YOLO باستخدام مسار الأوزان  
**English:** Load YOLO model using weights path

```python
        self.conf = conf
```
**العربية:** حفظ معدل الثقة كمتغير في الفئة  
**English:** Store confidence threshold as class variable

```python
        print(f"✅ Model: {weights_path}")
        print(f"✅ Device: {'CUDA' if self.device == 0 else 'CPU'}")
        print(f"📊 Confidence: {conf}")
```
**العربية:** طباعة معلومات النموذج والجهاز ومعدل الثقة للمستخدم  
**English:** Print model, device, and confidence information to user

---

### 4. دالة الكشف الأساسية | Main Detection Function

```python
    def detect(self, image_path):
```
**العربية:** تعريف دالة الكشف عن أسطوانات الغاز في الصورة  
**English:** Define function to detect gas cylinders in image

```python
        """Detect gas cylinders in image."""
```
**العربية:** وصف مختصر للدالة  
**English:** Brief description of the function

```python
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return []
```
**العربية:** التحقق من وجود ملف الصورة، وطباعة خطأ وإرجاع قائمة فارغة إذا لم توجد  
**English:** Check if image file exists, print error and return empty list if not found

```python
        print(f"🔍 Testing: {Path(image_path).name}")
```
**العربية:** طباعة اسم الملف الذي يتم اختباره  
**English:** Print name of file being tested

```python
        try:
            results = self.model(image_path, conf=self.conf, verbose=False)
```
**العربية:** محاولة تشغيل النموذج على الصورة مع معدل الثقة المحدد، وإخفاء التفاصيل  
**English:** Try running model on image with specified confidence, hiding verbose output

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
**العربية:** التكرار عبر كل صندوق مُكتشف والتحقق من أن ثقته أعلى من الحد الأدنى  
**English:** Loop through each detected box and check if confidence is above threshold

```python
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
```
**العربية:** استخراج إحداثيات الصندوق (x1,y1,x2,y2) وتحويلها لأرقام صحيحة  
**English:** Extract box coordinates (x1,y1,x2,y2) and convert to integers

```python
                        conf = box.conf.item()
```
**العربية:** استخراج قيمة الثقة للكشف  
**English:** Extract confidence value for the detection

```python
                        cylinders.append((x1, y1, x2, y2, conf))
```
**العربية:** إضافة الإحداثيات والثقة إلى قائمة الأسطوانات المُكتشفة  
**English:** Add coordinates and confidence to detected cylinders list

```python
            return cylinders
```
**العربية:** إرجاع قائمة الأسطوانات المُكتشفة  
**English:** Return list of detected cylinders

```python
        except Exception as e:
            print(f"❌ Error: {e}")
            return []
```
**العربية:** التعامل مع أي خطأ محتمل وطباعته وإرجاع قائمة فارغة  
**English:** Handle any potential errors, print them, and return empty list

---

### 5. دالة الرسم والتصور | Visualization Function

```python
    def visualize_results(self, image_path, cylinders):
```
**العربية:** تعريف دالة لرسم وعرض نتائج الكشف على الصورة  
**English:** Define function to draw and display detection results on image

```python
        """Create and save visualization."""
```
**العربية:** وصف مختصر للدالة  
**English:** Brief description of the function

```python
        try:
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
**العربية:** تحميل الصورة باستخدام OpenCV وتحويلها من BGR إلى RGB للعرض الصحيح  
**English:** Load image using OpenCV and convert from BGR to RGB for proper display

```python
            plt.figure(figsize=(12, 8))
            plt.imshow(img_rgb)
```
**العربية:** إنشاء شكل جديد بحجم 12x8 وعرض الصورة  
**English:** Create new figure with size 12x8 and display the image

```python
            for i, (x1, y1, x2, y2, conf) in enumerate(cylinders):
```
**العربية:** التكرار عبر كل أسطوانة مُكتشفة مع رقم تسلسلي  
**English:** Loop through each detected cylinder with sequential number

```python
                color = 'lime' if conf >= 0.7 else 'yellow' if conf >= 0.5 else 'orange'
```
**العربية:** تحديد لون الصندوق حسب مستوى الثقة:  
- أخضر فاتح للثقة العالية (≥70%)  
- أصفر للثقة المتوسطة (≥50%)  
- برتقالي للثقة المنخفضة  

**English:** Determine box color based on confidence level:  
- Lime for high confidence (≥70%)  
- Yellow for medium confidence (≥50%)  
- Orange for low confidence

```python
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color=color, linewidth=3)
                plt.gca().add_patch(rect)
```
**العربية:** إنشاء مستطيل حول الأسطوانة المُكتشفة بدون تعبئة وخط بسماكة 3  
**English:** Create rectangle around detected cylinder without fill and line width 3

```python
                plt.text(x1, y1-10, f'Cylinder {i+1}: {conf:.1%}', 
                        fontsize=12, color=color, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
```
**العربية:** إضافة نص يوضح رقم الأسطوانة ونسبة الثقة مع خلفية سوداء شفافة  
**English:** Add text showing cylinder number and confidence percentage with transparent black background

```python
            plt.title(f'Gas Cylinder Detection\n{Path(image_path).name}', 
                     fontsize=16, weight='bold')
```
**العربية:** إضافة عنوان للصورة يتضمن "كشف أسطوانات الغاز" واسم الملف  
**English:** Add title to image including "Gas Cylinder Detection" and filename

```python
            plt.axis('off')
```
**العربية:** إخفاء المحاور من الرسم البياني  
**English:** Hide axes from the plot

```python
            output_name = f"detection_{Path(image_path).stem}.png"
```
**العربية:** تحديد اسم ملف الإخراج باستخدام اسم الصورة الأصلية  
**English:** Set output filename using original image name

```python
            plt.savefig(output_name, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
```
**العربية:** حفظ الصورة بدقة 150 DPI وخلفية بيضاء وحدود محكمة  
**English:** Save image with 150 DPI resolution, white background, and tight bounds

```python
            print(f"💾 Saved: {output_name}")
            plt.show()
```
**العربية:** طباعة رسالة تأكيد الحفظ وعرض الصورة  
**English:** Print save confirmation message and display image

```python
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
```
**العربية:** التعامل مع أي خطأ في الرسم وطباعة رسالة تحذيرية  
**English:** Handle any visualization errors and print warning message

---

### 6. دالة الاختبار الرئيسية | Main Test Function

```python
    def test_image(self, image_path):
```
**العربية:** تعريف الدالة الرئيسية لاختبار صورة واحدة  
**English:** Define main function to test a single image

```python
        """Test single image and show results."""
```
**العربية:** وصف مختصر للدالة  
**English:** Brief description of the function

```python
        cylinders = self.detect(image_path)
```
**العربية:** استدعاء دالة الكشف للحصول على قائمة الأسطوانات المُكتشفة  
**English:** Call detection function to get list of detected cylinders

```python
        if cylinders:
            print(f"✅ Found {len(cylinders)} gas cylinder(s)!")
```
**العربية:** إذا وُجدت أسطوانات، طباعة رسالة نجاح مع العدد  
**English:** If cylinders found, print success message with count

```python
            for i, (x1, y1, x2, y2, conf) in enumerate(cylinders):
                print(f"  Cylinder {i+1}: {conf:.1%} confidence at ({x1},{y1})-({x2},{y2})")
```
**العربية:** طباعة تفاصيل كل أسطوانة: الرقم، الثقة، والإحداثيات  
**English:** Print details for each cylinder: number, confidence, and coordinates

```python
            self.visualize_results(image_path, cylinders)
```
**العربية:** استدعاء دالة الرسم لعرض النتائج بصرياً  
**English:** Call visualization function to display results visually

```python
        else:
            print("❌ No gas cylinders detected")
```
**العربية:** إذا لم تُوجد أسطوانات، طباعة رسالة عدم وجود كشف  
**English:** If no cylinders found, print no detection message

```python
        return cylinders
```
**العربية:** إرجاع قائمة الأسطوانات للاستخدام خارج الدالة  
**English:** Return cylinders list for external use

---

### 7. الدالة الرئيسية | Main Function

```python
def main():
```
**العربية:** تعريف الدالة الرئيسية للبرنامج  
**English:** Define main program function

```python
    parser = argparse.ArgumentParser(description="Gas cylinder single image tester")
```
**العربية:** إنشاء كائن لتحليل معاملات سطر الأوامر مع وصف البرنامج  
**English:** Create argument parser object with program description

```python
    parser.add_argument("--weights", default="runs/train/cylinder_detector/weights/best.pt", 
                       help="Model weights path")
```
**العربية:** إضافة معامل لمسار أوزان النموذج مع قيمة افتراضية  
**English:** Add argument for model weights path with default value

```python
    parser.add_argument("--image", required=True, help="Image path to test")
```
**العربية:** إضافة معامل إجباري لمسار الصورة المراد اختبارها  
**English:** Add required argument for image path to test

```python
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
```
**العربية:** إضافة معامل لحد الثقة كرقم عشري مع قيمة افتراضية 0.5  
**English:** Add argument for confidence threshold as float with default 0.5

```python
    args = parser.parse_args()
```
**العربية:** تحليل معاملات سطر الأوامر المُدخلة  
**English:** Parse input command-line arguments

```python
    try:
        tester = GasCylinderTester(args.weights, args.conf)
        tester.test_image(args.image)
```
**العربية:** محاولة إنشاء كائن الاختبار وتشغيل اختبار الصورة  
**English:** Try creating tester object and running image test

```python
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
```
**العربية:** التعامل مع إيقاف البرنامج بواسطة المستخدم (Ctrl+C)  
**English:** Handle program interruption by user (Ctrl+C)

```python
    except Exception as e:
        print(f"❌ Error: {e}")
```
**العربية:** التعامل مع أي خطأ آخر وطباعته  
**English:** Handle any other errors and print them

```python
if __name__ == "__main__":
    main()
```
**العربية:** تشغيل الدالة الرئيسية فقط إذا تم تشغيل الملف مباشرة (وليس كمكتبة)  
**English:** Run main function only if file is executed directly (not imported as library)

---

## خلاصة البرنامج | Program Summary

### الغرض | Purpose
**العربية:** هذا البرنامج مخصص لكشف أسطوانات الغاز في الصور المفردة باستخدام نموذج YOLO المُدرب، ويعرض النتائج بصرياً مع معلومات مفصلة عن كل كشف.

**English:** This program is designed to detect gas cylinders in single images using a trained YOLO model, displaying results visually with detailed information about each detection.

### كيفية الاستخدام | How to Use
```bash
python src/test_single_image.py --image "path/to/image.jpg"
python src/test_single_image.py --image "path/to/image.jpg" --conf 0.7 --weights "custom_model.pt"
```

### المخرجات | Outputs
**العربية:**
- رسائل نصية تُظهر نتائج الكشف
- صورة بصرية مع مربعات حول الأسطوانات المُكتشفة
- ملف صورة محفوظ يحتوي على النتائج

**English:**
- Text messages showing detection results
- Visual image with boxes around detected cylinders  
- Saved image file containing the results