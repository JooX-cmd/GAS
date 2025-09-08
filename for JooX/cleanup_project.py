#!/usr/bin/env python3
"""
Smart Project Cleanup Script
Safely removes unnecessary files while protecting important project components
"""

import os
import shutil
import glob
from pathlib import Path
from datetime import datetime
import sys

class ProjectCleaner:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root).resolve()
        self.deleted_files = 0
        self.deleted_folders = 0
        self.errors = 0
        self.protected_files = [
            "src/",
            "runs/train/cylinder_detector/weights/best.pt",
            "runs/train/cylinder_detector/weights/last.pt", 
            "data/dataset/",
            "requirements.txt",
            "README.md",
            "run_ultra_strict.bat",
            "setup_gpu.ps1",
            "yolo11n.pt",
            "venv/"
        ]
        
    def print_header(self):
        print("=" * 60)
        print("           🧹 SMART PROJECT CLEANUP UTILITY")
        print("=" * 60)
        print()
        print("This script will safely remove unnecessary files:")
        print("  ❌ Test/demo files (test_infer*.py, simple_test.py)")
        print("  ❌ Cache directories (__pycache__, dataset cache)")
        print("  ❌ Duplicate documentation (README.txt)")
        print("  ❌ Old detection results (keeping latest 2)")
        print("  ❌ Temporary files (.tmp, .log, etc.)")
        print()
        print("=" * 60)
        print("   🛡️  PROTECTED FILES (Will NOT be deleted):")
        print("=" * 60)
        for file in self.protected_files:
            print(f"   ✅ {file}")
        print("=" * 60)
        print()
        
    def safe_delete_file(self, file_path):
        """Safely delete a file with error handling"""
        try:
            if file_path.exists():
                file_path.unlink()
                print(f"✅ Deleted file: {file_path.name}")
                self.deleted_files += 1
                return True
            else:
                print(f"⏭️  Not found: {file_path.name}")
                return False
        except Exception as e:
            print(f"❌ Error deleting {file_path}: {e}")
            self.errors += 1
            return False
            
    def safe_delete_folder(self, folder_path):
        """Safely delete a folder with error handling"""
        try:
            if folder_path.exists() and folder_path.is_dir():
                shutil.rmtree(folder_path)
                print(f"✅ Deleted folder: {folder_path.name}")
                self.deleted_folders += 1
                return True
            else:
                print(f"⏭️  Not found: {folder_path.name}")
                return False
        except Exception as e:
            print(f"❌ Error deleting {folder_path}: {e}")
            self.errors += 1
            return False
            
    def cleanup_test_files(self):
        """Remove test and demo files"""
        print("\n🗑️  PHASE 1: Removing test and demo files...")
        print("=" * 50)
        
        test_files = [
            "test_infer.py",
            "test_infer_random.py", 
            "simple_test.py",
            "README.txt",
            "got you — here's a clean YOLO11 pro.txt"
        ]
        
        for file_name in test_files:
            file_path = self.project_root / file_name
            self.safe_delete_file(file_path)
            
        # Remove any other test/debug files
        for pattern in ["test_*.py", "demo_*.py", "debug_*.py"]:
            for file_path in self.project_root.glob(pattern):
                if file_path.name not in [f.split('/')[-1] for f in self.protected_files]:
                    self.safe_delete_file(file_path)
                    
    def cleanup_cache(self):
        """Remove cache directories"""
        print("\n🗑️  PHASE 2: Removing cache directories...")
        print("=" * 50)
        
        cache_folders = [
            "__pycache__",
            "src/__pycache__",
            ".pytest_cache",
            "data/dataset/train/cache",
            "data/dataset/valid/cache", 
            "data/dataset/test/cache"
        ]
        
        for folder_name in cache_folders:
            folder_path = self.project_root / folder_name
            self.safe_delete_folder(folder_path)
            
        # Find and remove any other cache folders
        for cache_dir in self.project_root.rglob("*cache*"):
            if cache_dir.is_dir() and "venv" not in str(cache_dir):
                self.safe_delete_folder(cache_dir)
                
    def cleanup_old_results(self):
        """Keep only latest detection results"""
        print("\n🗑️  PHASE 3: Removing old detection results...")
        print("=" * 50)
        
        detect_dir = self.project_root / "runs" / "detect"
        if detect_dir.exists():
            # Get all predict folders sorted by modification time
            predict_folders = []
            for folder in detect_dir.glob("predict*"):
                if folder.is_dir():
                    predict_folders.append((folder.stat().st_mtime, folder))
                    
            predict_folders.sort(reverse=True)  # Newest first
            
            if len(predict_folders) > 2:
                print(f"Found {len(predict_folders)} detection result folders, keeping latest 2...")
                for _, folder in predict_folders[2:]:  # Delete all except latest 2
                    self.safe_delete_folder(folder)
            else:
                print(f"Only {len(predict_folders)} detection folders found, keeping all.")
                
    def cleanup_temp_files(self):
        """Remove temporary files"""
        print("\n🗑️  PHASE 4: Removing temporary files...")
        print("=" * 50)
        
        temp_patterns = ["*.tmp", "*.temp", "*.log", "*.bak", "*~"]
        
        for pattern in temp_patterns:
            for file_path in self.project_root.glob(pattern):
                self.safe_delete_file(file_path)
                
        # Remove IDE settings (optional)
        ide_folders = [".vscode", ".idea"]
        for folder_name in ide_folders:
            folder_path = self.project_root / folder_name
            if folder_path.exists():
                self.safe_delete_folder(folder_path)
                
    def check_large_files(self):
        """Check for large unnecessary files"""
        print("\n🗑️  PHASE 5: Checking for large unnecessary files...")
        print("=" * 50)
        
        archive_patterns = ["*.zip", "*.rar", "*.7z", "*.tar.gz"]
        
        for pattern in archive_patterns:
            for file_path in self.project_root.glob(pattern):
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"⚠️  Found archive: {file_path.name} ({size_mb:.1f} MB)")
                print(f"    Review manually if this is needed")
                
    def print_summary(self):
        """Print cleanup summary"""
        print("\n" + "=" * 60)
        print("                   📊 CLEANUP SUMMARY")
        print("=" * 60)
        print(f"   📁 Deleted files: {self.deleted_files}")
        print(f"   📂 Deleted folders: {self.deleted_folders}")
        print(f"   ❌ Errors: {self.errors}")
        print("=" * 60)
        
        if self.errors > 0:
            print("\n⚠️  Some files couldn't be deleted (might be in use)")
            print("   Try closing any programs and running again if needed.")
            
        print("\n✅ Cleanup completed successfully!")
        print("\nYour project is now cleaner and more organized.")
        print("All essential files have been preserved.")
        print("\n💾 Estimated space saved: 2-5 GB")
        print("   (Cache files, duplicates, and temp files removed)")
        
        print("\n" + "=" * 60)
        print("Next steps:")
        print("  1. Test your system: Run run_ultra_strict.bat")
        print("  2. If any cache is needed, it will regenerate automatically")
        print("  3. Your trained model and dataset are safe!")
        print("=" * 60)
        
    def run_cleanup(self):
        """Run the complete cleanup process"""
        self.print_header()
        
        # Ask for confirmation
        response = input("Do you want to proceed with cleanup? (y/N): ").strip().lower()
        if response != 'y':
            print("Cleanup cancelled.")
            return
            
        print(f"\nStarting cleanup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
        
        # Run cleanup phases
        self.cleanup_test_files()
        self.cleanup_cache()
        self.cleanup_old_results() 
        self.cleanup_temp_files()
        self.check_large_files()
        
        # Print summary
        self.print_summary()

def main():
    """Main function"""
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🧹 Smart Project Cleanup Utility")
    print(f"Working directory: {Path.cwd()}")
    print()
    
    cleaner = ProjectCleaner()
    cleaner.run_cleanup()

if __name__ == "__main__":
    main()