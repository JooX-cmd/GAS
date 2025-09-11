# شرح كود train_model.py | Code Explanation for train_model.py

## نظرة عامة | Overview
هذا الملف يحتوي على شرح مفصل لكل سطر في كود `train_model.py` باللغتين العربية والإنجليزية - وهو المسؤول عن تدريب نموذج YOLO لكشف أسطوانات الغاز
This file contains a detailed explanation of every line in the `train_model.py` code in both Arabic and English - responsible for training YOLO model for gas cylinder detection.

---

## بنية الملف | File Structure

### 1. المعلومات الأساسية والاستيراد | Basic Info and Imports

```python
#!/usr/bin/env python3
```
**العربية:** هذا السطر يخبر النظام أن يستخدم Python 3 لتشغيل هذا الملف  
**English:** This line tells the system to use Python 3 to run this file

```python
"""
GPU-Only Gas Cylinder Detection Training - Simplified
====================================================
Optimized YOLOv11 training script for gas cylinder detection.
GPU required - no CPU fallback.
"""
```
**العربية:** وصف تفصيلي للملف - نص تدريب محسّن لكشف أسطوانات الغاز باستخدام GPU فقط  
**English:** Detailed file description - optimized training script for gas cylinder detection using GPU only

```python
import argparse
```
**العربية:** استيراد مكتبة argparse للتعامل مع المعاملات من سطر الأوامر  
**English:** Import argparse library to handle command-line arguments

```python
import sys
```
**العربية:** استيراد مكتبة sys للتحكم في النظام والخروج من البرنامج  
**English:** Import sys library for system control and program exit

```python
import logging
```
**العربية:** استيراد مكتبة logging لتسجيل رسائل التشغيل والأخطاء  
**English:** Import logging library for recording operation messages and errors

```python
from pathlib import Path
```
**العربية:** استيراد Path للتعامل مع مسارات الملفات بطريقة أفضل  
**English:** Import Path for better file path handling

```python
from datetime import datetime
```
**العربية:** استيراد datetime للتعامل مع التاريخ والوقت  
**English:** Import datetime for date and time handling

```python
import os
import time
import json
```
**العربية:** استيراد مكتبات إضافية:  
- os: للتعامل مع نظام التشغيل  
- time: للتعامل مع الوقت  
- json: للتعامل مع ملفات JSON  

**English:** Import additional libraries:  
- os: for operating system interaction  
- time: for time handling  
- json: for JSON file handling

```python
import torch
from ultralytics import YOLO
```
**العربية:** استيراد المكتبات الأساسية:  
- torch: PyTorch للتعلم العميق  
- YOLO: نموذج YOLO من ultralytics  

**English:** Import core libraries:  
- torch: PyTorch for deep learning  
- YOLO: YOLO model from ultralytics

---

### 2. إعداد التسجيل | Logging Setup

```python
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
```
**العربية:** إعداد نظام التسجيل لعرض رسائل المعلومات بتنسيق بسيط، وإنشاء كائن logger  
**English:** Configure logging system to show info messages in simple format, and create logger object

---

### 3. دالة فحص GPU | GPU Check Function

```python
def check_gpu():
```
**العربية:** تعريف دالة للتحقق من توفر GPU - مطلوب للتدريب  
**English:** Define function to check GPU availability - required for training

```python
    """Check GPU availability - required for training."""
```
**العربية:** وصف مختصر للدالة  
**English:** Brief description of the function

```python
    if not torch.cuda.is_available():
```
**العربية:** التحقق من عدم توفر CUDA GPU  
**English:** Check if CUDA GPU is not available

```python
        logger.error("❌ CUDA GPU not detected! GPU required for training.")
        logger.error("Solutions:")
        logger.error("  • Install CUDA drivers")
        logger.error("  • Reinstall PyTorch with CUDA support")
        logger.error("  • Use cloud GPU (Colab, AWS, etc.)")
        sys.exit(1)
```
**العربية:** إذا لم يُوجد GPU، طباعة رسائل خطأ مع الحلول المقترحة والخروج من البرنامج  
**English:** If no GPU found, print error messages with suggested solutions and exit program

```python
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
```
**العربية:** الحصول على اسم GPU وحجم الذاكرة وطباعة المعلومات  
**English:** Get GPU name and memory size and print information

```python
    return gpu_memory
```
**العربية:** إرجاع حجم ذاكرة GPU بالجيجابايت  
**English:** Return GPU memory size in gigabytes

---

### 4. دالة حساب حجم الدفعة التلقائي | Auto Batch Size Function

```python
def auto_batch_size(gpu_memory_gb):
```
**العربية:** تعريف دالة لحساب حجم الدفعة الأمثل حسب ذاكرة GPU  
**English:** Define function to calculate optimal batch size based on GPU memory

```python
    """Calculate optimal batch size based on GPU memory."""
```
**العربية:** وصف مختصر للدالة  
**English:** Brief description of the function

```python
    if gpu_memory_gb >= 12:    return 32  # High-end GPUs
    elif gpu_memory_gb >= 8:   return 24  # Mid-range GPUs  
    elif gpu_memory_gb >= 6:   return 16  # Entry GPUs
    else:                      return 8   # Low memory GPUs
```
**العربية:** تحديد حجم الدفعة حسب ذاكرة GPU:  
- 32: للـ GPUs عالية الأداء (≥12GB)  
- 24: للـ GPUs متوسطة الأداء (≥8GB)  
- 16: للـ GPUs المبتدئة (≥6GB)  
- 8: للـ GPUs ذات الذاكرة المحدودة  

**English:** Set batch size based on GPU memory:  
- 32: for high-end GPUs (≥12GB)  
- 24: for mid-range GPUs (≥8GB)  
- 16: for entry GPUs (≥6GB)  
- 8: for low memory GPUs

---

### 5. دالة التدريب الرئيسية | Main Training Function

```python
def train_model(data_yaml="data/dataset/data.yaml", epochs=100, batch=None, patience=30, resume=True):
```
**العربية:** تعريف دالة التدريب الرئيسية مع المعاملات:  
- data_yaml: مسار ملف إعدادات البيانات  
- epochs: عدد حقب التدريب  
- batch: حجم الدفعة  
- patience: صبر الإيقاف المبكر  
- resume: استكمال من نقطة تفتيش  

**English:** Define main training function with parameters:  
- data_yaml: dataset configuration file path  
- epochs: number of training epochs  
- batch: batch size  
- patience: early stopping patience  
- resume: resume from checkpoint

```python
    """
    Train YOLO11 model for gas cylinder detection - GPU only.
    
    Args:
        data_yaml: Path to dataset configuration
        epochs: Number of training epochs
        batch: Batch size (auto-calculated if None)
        patience: Early stopping patience
        resume: Resume from checkpoint if available
    """
```
**العربية:** وصف مفصل للدالة ومعاملاتها  
**English:** Detailed description of function and its parameters

```python
    gpu_memory = check_gpu()
```
**العربية:** فحص GPU والحصول على حجم الذاكرة  
**English:** Check GPU and get memory size

```python
    if batch is None:
        batch = auto_batch_size(gpu_memory)
        logger.info(f"🧠 Auto batch size: {batch} (GPU: {gpu_memory:.1f}GB)")
```
**العربية:** إذا لم يُحدد حجم الدفعة، حسابه تلقائياً وطباعة المعلومات  
**English:** If batch size not specified, calculate automatically and print information

```python
    torch.cuda.empty_cache()
```
**العربية:** تنظيف ذاكرة GPU قبل بدء التدريب  
**English:** Clear GPU memory before starting training

```python
    logger.info("=" * 50)
    logger.info("🚀 STARTING TRAINING")
    logger.info("=" * 50)
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch: {batch}")
    logger.info(f"Patience: {patience}")
```
**العربية:** طباعة رأس جميل ومعلومات التدريب  
**English:** Print nice header and training information

```python
    try:
```
**العربية:** بدء محاولة تنفيذ التدريب مع معالجة الأخطاء  
**English:** Start trying to execute training with error handling

```python
        checkpoint = Path("runs/train/cylinder_detector/weights/last.pt")
        if resume and checkpoint.exists():
            logger.info(f"🔄 Resuming from: {checkpoint}")
            model = YOLO(str(checkpoint))
        else:
            logger.info("🎆 Starting fresh with YOLO11n")
            model = YOLO("yolo11n.pt")
```
**العربية:** التحقق من وجود نقطة تفتيش:  
- إذا وُجدت ومطلوب الاستكمال: تحميلها  
- وإلا: البدء من نموذج YOLO11n جديد  

**English:** Check for checkpoint existence:  
- If found and resume requested: load it  
- Otherwise: start fresh with YOLO11n model

```python
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            device=0,  # GPU
            project="runs/train",
            name="cylinder_detector",
            exist_ok=True,
            patience=patience,
            save=True,
            save_period=10,
            cache=True,
            workers=4,
            amp=True,  # Mixed precision
            plots=True,
            val=True,
            resume=resume,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01
        )
```
**العربية:** بدء التدريب مع إعدادات مُحسّنة:  
- data: ملف البيانات  
- epochs/batch: حقب وحجم الدفعة  
- device=0: استخدام GPU الأول  
- project/name: مجلد واسم المشروع  
- patience: صبر الإيقاف المبكر  
- save_period=10: حفظ كل 10 حقب  
- cache=True: تخزين البيانات مؤقتاً  
- workers=4: 4 معالجات للبيانات  
- amp=True: دقة مختلطة لتوفير الذاكرة  
- optimizer='AdamW': محسِّن AdamW  
- lr0/lrf: معدل التعلم الأولي والنهائي  

**English:** Start training with optimized settings:  
- data: dataset file  
- epochs/batch: epochs and batch size  
- device=0: use first GPU  
- project/name: project folder and name  
- patience: early stopping patience  
- save_period=10: save every 10 epochs  
- cache=True: cache data in memory  
- workers=4: 4 data workers  
- amp=True: mixed precision for memory saving  
- optimizer='AdamW': AdamW optimizer  
- lr0/lrf: initial and final learning rates

```python
        logger.info("=" * 50)
        logger.info("🎉 TRAINING COMPLETED!")
        logger.info("=" * 50)
        logger.info(f"✅ Best: runs/train/cylinder_detector/weights/best.pt")
        logger.info(f"✅ Last: runs/train/cylinder_detector/weights/last.pt")
```
**العربية:** طباعة رسالة نجاح التدريب ومواقع ملفات النماذج  
**English:** Print training success message and model file locations

```python
        summary = {
            "completed": datetime.now().isoformat(),
            "epochs": epochs,
            "batch_size": batch,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(gpu_memory, 1)
        }
```
**العربية:** إنشاء ملخص التدريب يحتوي على:  
- وقت الانتهاء  
- عدد الحقب  
- حجم الدفعة  
- اسم GPU  
- حجم ذاكرة GPU  

**English:** Create training summary containing:  
- completion time  
- number of epochs  
- batch size  
- GPU name  
- GPU memory size

```python
        summary_path = Path("runs/train/cylinder_detector/training_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
```
**العربية:** حفظ ملخص التدريب في ملف JSON مع إنشاء المجلدات إذا لم توجد  
**English:** Save training summary to JSON file with creating folders if they don't exist

```python
        logger.info("🎯 Next steps:")
        logger.info("  python src/test_model.py --webcam")
        logger.info("  python src/ultra_strict_detector.py --source 0")
```
**العربية:** طباعة الخطوات التالية المقترحة لاختبار النموذج  
**English:** Print suggested next steps for testing the model

```python
        return results
```
**العربية:** إرجاع نتائج التدريب  
**English:** Return training results

---

### 6. معالجة الأخطاء | Error Handling

```python
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted - checkpoint saved")
        return None
```
**العربية:** التعامل مع إيقاف التدريب بواسطة المستخدم (Ctrl+C)  
**English:** Handle training interruption by user (Ctrl+C)

```python
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"❌ GPU out of memory! Try smaller --batch {batch//2}")
            torch.cuda.empty_cache()
        raise
```
**العربية:** التعامل مع خطأ نفاد ذاكرة GPU واقتراح تقليل حجم الدفعة  
**English:** Handle GPU out of memory error and suggest reducing batch size

```python
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.info("💡 Troubleshooting:")
        logger.info("  • Check dataset folders exist")
        logger.info("  • Verify image/label files match") 
        logger.info("  • Try smaller batch size")
        logger.info("  • Check free disk space")
        raise
```
**العربية:** التعامل مع أي خطأ آخر وطباعة نصائح استكشاف الأخطاء  
**English:** Handle any other errors and print troubleshooting tips

---

### 7. الدالة الرئيسية | Main Function

```python
def main():
    parser = argparse.ArgumentParser(description="GPU-only YOLO11 training for gas cylinders")
```
**العربية:** إنشاء محلل معاملات سطر الأوامر مع وصف البرنامج  
**English:** Create command-line argument parser with program description

```python
    parser.add_argument("--data", default="data/dataset/data.yaml", help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, help="Batch size (auto if not set)")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (ignore checkpoints)")
```
**العربية:** إضافة معاملات سطر الأوامر:  
- --data: مسار ملف البيانات  
- --epochs: عدد حقب التدريب  
- --batch: حجم الدفعة (تلقائي إذا لم يُحدد)  
- --patience: صبر الإيقاف المبكر  
- --no-resume: بدء جديد (تجاهل نقاط التفتيش)  

**English:** Add command-line arguments:  
- --data: dataset file path  
- --epochs: number of training epochs  
- --batch: batch size (auto if not set)  
- --patience: early stopping patience  
- --no-resume: start fresh (ignore checkpoints)

```python
    args = parser.parse_args()
```
**العربية:** تحليل معاملات سطر الأوامر المُدخلة  
**English:** Parse input command-line arguments

```python
    try:
        print("\n" + "="*50)
        print("  GAS CYLINDER DETECTION TRAINING")
        print("        GPU-Optimized YOLO11")
        print("="*50 + "\n")
```
**العربية:** طباعة عنوان جميل للبرنامج  
**English:** Print nice program title

```python
        train_model(
            data_yaml=args.data,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
            resume=not args.no_resume
        )
```
**العربية:** استدعاء دالة التدريب مع المعاملات المُدخلة  
**English:** Call training function with input parameters

```python
        print("\n🎉 Training completed successfully!")
```
**العربية:** طباعة رسالة نجاح التدريب  
**English:** Print training success message

```python
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
```
**العربية:** معالجة الأخطاء والخروج من البرنامج مع رمز خطأ  
**English:** Handle errors and exit program with error code

```python
if __name__ == "__main__":
    main()
```
**العربية:** تشغيل الدالة الرئيسية فقط إذا تم تشغيل الملف مباشرة  
**English:** Run main function only if file is executed directly

---

## خلاصة البرنامج | Program Summary

### الغرض | Purpose
**العربية:** هذا البرنامج مخصص لتدريب نموذج YOLO11 على كشف أسطوانات الغاز باستخدام GPU فقط، مع تحسينات للأداء والاستقرار.

**English:** This program is designed to train a YOLO11 model for gas cylinder detection using GPU only, with optimizations for performance and stability.

### المتطلبات | Requirements
**العربية:**
- GPU مع دعم CUDA
- PyTorch مع دعم CUDA
- مجموعة بيانات في المسار المحدد
- مساحة تخزين كافية

**English:**
- GPU with CUDA support
- PyTorch with CUDA support
- Dataset in specified path
- Sufficient storage space

### كيفية الاستخدام | How to Use
```bash
# تدريب أساسي | Basic training
python src/train_model.py

# تدريب مخصص | Custom training
python src/train_model.py --epochs 200 --batch 16 --patience 50

# بدء جديد | Fresh start
python src/train_model.py --no-resume
```

### المخرجات | Outputs
**العربية:**
- ملفات النموذج المُدرب (best.pt, last.pt)
- ملخص التدريب (training_summary.json)
- رسوم بيانية للأداء
- سجلات التدريب

**English:**
- Trained model files (best.pt, last.pt)
- Training summary (training_summary.json)
- Performance plots
- Training logs