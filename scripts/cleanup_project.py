import os
import shutil

"""
Urea AI Project - Workspace Cleanup Script
------------------------------------------
Task: Archives intermediate ML artifacts to prepare for C++ Firmware phase.
Keeps: Source code (.py, .cpp, .ino, .h), TFLite models, and core dataset.
"""

def cleanup_project():
    # Configuration
    archive_folder = "Archive_Old_ML"
    keep_exts = {".py", ".cpp", ".ino", ".h", ".tflite"}
    keep_files = {"milk_combined_full_dataset.csv"}
    
    # Ensure current script doesn't move itself
    script_name = os.path.basename(__file__)
    
    if not os.path.exists(archive_folder):
        os.makedirs(archive_folder)
        print(f"Created archive folder: {archive_folder}")

    files_moved = []
    files_kept = []

    for item in os.listdir("."):
        # Process files only (skip folders like .venv and Archive_Old_ML)
        if os.path.isfile(item):
            ext = os.path.splitext(item)[1].lower()
            
            # Determine if we keep the file
            if ext in keep_exts or item in keep_files or item == script_name:
                files_kept.append(item)
            else:
                try:
                    shutil.move(item, os.path.join(archive_folder, item))
                    files_moved.append(item)
                except Exception as e:
                    print(f"Error moving {item}: {e}")

    # Print Clean Summary
    print("\n" + "="*40)
    print("      PROJECT CLEANUP SUMMARY")
    print("="*40)
    
    print(f"\nKEPT CORE ASSETS ({len(files_kept)}):")
    for f in sorted(files_kept):
        print(f"  [+] {f}")
        
    print(f"\nMOVED TO ARCHIVE ({len(files_moved)}):")
    for f in sorted(files_moved):
        print(f"  [->] {f}")
        
    print("\n" + "="*40)
    print(f"Cleanup complete. Workspace optimized for Firmware Phase.")

if __name__ == "__main__":
    cleanup_project()
