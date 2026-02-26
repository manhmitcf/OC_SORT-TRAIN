import os
import shutil
import sys
from roboflow import Roboflow

# Add current directory to path to allow importing the converter
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from convert_roboflow_to_mot import convert_roboflow_to_mot
except ImportError:
    print("Error: Could not import convert_roboflow_to_mot. Make sure tools/convert_roboflow_to_mot.py exists.")
    sys.exit(1)

def download_and_prepare():
    print("--- 1. Downloading Dataset from Roboflow (COCO format) ---")
    try:
        rf = Roboflow(api_key="7QSUEERZ8yV6mIjL8oiv")
        project = rf.workspace("ok-vblps").project("dat_labeling")
        version = project.version(8)
        dataset = version.download("coco")
    except Exception as e:
        print(f"Error downloading from Roboflow: {e}")
        print("Please ensure you have installed roboflow: pip install roboflow")
        return

    download_dir = dataset.location
    print(f"Dataset downloaded to temporary location: {download_dir}")

    # Define target root
    # Go up one level from tools/ to project root, then into datasets/custom_dataset
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_root = os.path.join(project_root, "datasets", "custom_dataset")

    print(f"--- 2. Organizing data into {target_root} ---")
    
    if os.path.exists(target_root):
        print(f"Cleaning up existing directory: {target_root}")
        shutil.rmtree(target_root)
    os.makedirs(target_root)
    os.makedirs(os.path.join(target_root, "annotations"), exist_ok=True)

    # Map Roboflow splits to our folder names
    # Roboflow usually gives: train, valid, test
    split_map = {
        'train': 'train',
        'valid': 'val',
        'test': 'test'
    }

    for rf_split, target_split in split_map.items():
        src_split_dir = os.path.join(download_dir, rf_split)
        
        # Check if this split exists in the download
        if not os.path.exists(src_split_dir):
            continue
            
        print(f"Processing split: {rf_split} -> {target_split}")
        
        target_img_dir = os.path.join(target_root, target_split)
        os.makedirs(target_img_dir, exist_ok=True)

        # Move files
        files = os.listdir(src_split_dir)
        json_file = None
        
        for f in files:
            src_path = os.path.join(src_split_dir, f)
            
            if f.endswith(".json") or f == "_annotations.coco.json":
                json_file = f
            elif f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                shutil.move(src_path, os.path.join(target_img_dir, f))

        # Handle Annotation
        if json_file:
            src_json_path = os.path.join(src_split_dir, json_file)
            target_json_name = f"{target_split}.json"
            target_json_path = os.path.join(target_root, "annotations", target_json_name)
            
            # Move the raw COCO json
            shutil.move(src_json_path, target_json_path)
            
            # Convert to MOT format
            target_mot_json_path = os.path.join(target_root, "annotations", f"{target_split}_mot.json")
            print(f"  Converting annotations to MOT format: {target_mot_json_path}")
            convert_roboflow_to_mot(target_json_path, target_mot_json_path)
        else:
            print(f"  Warning: No JSON annotation found for {rf_split}")

    # Cleanup download dir if empty or needed
    try:
        shutil.rmtree(download_dir)
        print("Removed temporary download directory.")
    except:
        pass

    print("\n--- Done! ---")
    print(f"Data is ready at: {target_root}")
    print("You can now run training using the instructions in README_custom_train.md")

if __name__ == "__main__":
    download_and_prepare()
