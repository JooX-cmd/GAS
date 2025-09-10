#!/usr/bin/env python3
"""
Quick test to verify the Gas Cylinder Detection system is working
"""
import sys
import os

def test_imports():
    print("🔍 Testing imports...")
    try:
        import torch
        print(f"  ✅ PyTorch {torch.__version__}")
        
        import cv2
        print(f"  ✅ OpenCV {cv2.__version__}")
        
        import numpy
        print(f"  ✅ NumPy {numpy.__version__}")
        
        from ultralytics import YOLO
        print(f"  ✅ Ultralytics YOLO")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_model_files():
    print("🔍 Testing model files...")
    
    base_model = "yolo11n.pt"
    if os.path.exists(base_model):
        print(f"  ✅ Base model found: {base_model}")
    else:
        print(f"  ❌ Base model missing: {base_model}")
        
    trained_model = "runs/train/cylinder_detector/weights/best.pt"
    if os.path.exists(trained_model):
        print(f"  ✅ Trained model found: {trained_model}")
        return True
    else:
        print(f"  ❌ Trained model missing: {trained_model}")
        return False

def test_scripts():
    print("🔍 Testing Python scripts...")
    scripts = [
        "src/train_model.py",
        "src/test_model.py", 
        "src/ultra_strict_detector.py"
    ]
    
    all_exist = True
    for script in scripts:
        if os.path.exists(script):
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script}")
            all_exist = False
    
    return all_exist

def test_dataset():
    print("🔍 Testing dataset...")
    
    data_yaml = "data/dataset/data.yaml"
    if os.path.exists(data_yaml):
        print(f"  ✅ Dataset config found: {data_yaml}")
        return True
    else:
        print(f"  ❌ Dataset config missing: {data_yaml}")
        return False

def main():
    print("🚀 Gas Cylinder Detection System Test")
    print("=" * 50)
    
    # Change to the right directory if needed
    if os.path.basename(os.getcwd()) == 'src':
        os.chdir('..')
    
    tests = [
        ("Imports", test_imports),
        ("Model Files", test_model_files),
        ("Scripts", test_scripts),
        ("Dataset", test_dataset)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append(result)
        print()
    
    print("=" * 50)
    if all(results):
        print("🎉 ALL TESTS PASSED! Your system is ready to use!")
        print("\n💡 Quick start:")
        print("   1. Double-click 'run_ultra_strict.bat' for easy detection")
        print("   2. Or run: python src/ultra_strict_detector.py --source 0")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    print("\n🔧 Available scripts:")
    print("   • Train model: python src/train_model.py --data data/dataset/data.yaml")
    print("   • Test image: python src/test_model.py --image path/to/image.jpg")
    print("   • Live detection: python src/ultra_strict_detector.py --source 0")

if __name__ == "__main__":
    main()
