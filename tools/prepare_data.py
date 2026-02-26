import os
import shutil
import sys
import json
import re
from roboflow import Roboflow

# ==============================================================================
# HÀM CONVERT ĐƯỢC NHÚNG TRỰC TIẾP VÀO SCRIPT ĐỂ CHẠY ĐỘC LẬP
# ==============================================================================
def convert_roboflow_to_mot(json_path, output_path):
    """
    Chuyển đổi file COCO JSON từ Roboflow sang định dạng mà OC-SORT yêu cầu.
    - Chỉ giữ lại class 'person' và map id về 1.
    - Thêm các trường 'video_id', 'frame_id', 'track_id' nếu thiếu.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Tìm và lọc category 'person'
    person_cat_id = None
    for cat in data['categories']:
        if cat['name'] == 'person':
            person_cat_id = cat['id']
            break
    
    if person_cat_id is None:
        print(f"Lỗi: Không tìm thấy category 'person' trong file {json_path}")
        return

    new_categories = [{"id": 1, "name": "person", "supercategory": "none"}]
    
    # 2. Cập nhật thông tin ảnh, thêm video_id và frame_id
    new_images = []
    for i, img in enumerate(data['images']):
        new_img = img.copy()
        if 'video_id' not in new_img:
            new_img['video_id'] = 1
        if 'frame_id' not in new_img:
            try:
                match = re.search(r'frame_(\d+)', new_img['file_name'])
                if match:
                    new_img['frame_id'] = int(match.group(1))
                else:
                    new_img['frame_id'] = i + 1
            except:
                new_img['frame_id'] = i + 1
        new_images.append(new_img)

    # 3. Cập nhật annotations, đổi category_id và thêm track_id
    new_annotations = []
    for ann in data['annotations']:
        if ann['category_id'] == person_cat_id:
            new_ann = ann.copy()
            new_ann['category_id'] = 1  # Ép id về 1
            if 'track_id' not in new_ann:
                new_ann['track_id'] = -1
            new_annotations.append(new_ann)

    # 4. Tạo file JSON mới
    new_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": new_categories,
        "images": new_images,
        "annotations": new_annotations
    }

    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"  => Đã chuyển đổi thành công: {output_path}")

# ==============================================================================
# LOGIC CHÍNH ĐỂ TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ==============================================================================
def download_and_prepare():
    print("--- 1. Tải dữ liệu từ Roboflow (định dạng COCO) ---")
    try:
        # Thay thế bằng API key của bạn nếu cần
        rf = Roboflow(api_key="7QSUEERZ8yV6mIjL8oiv")
        project = rf.workspace("ok-vblps").project("dat_labeling")
        version = project.version(10)
        # Tải về với định dạng COCO
        dataset = version.download("coco")
    except Exception as e:
        print(f"Lỗi khi tải từ Roboflow: {e}")
        print("Hãy đảm bảo bạn đã cài đặt roboflow: !pip install roboflow")
        return

    download_dir = dataset.location
    print(f"Dữ liệu đã được tải về thư mục tạm: {download_dir}")

    # Trong Kaggle, chúng ta làm việc tại /kaggle/working/OC_SORT
    # Thư mục đích sẽ là 'datasets/custom_dataset'
    target_root = "datasets/custom_dataset"

    print(f"--- 2. Sắp xếp dữ liệu vào thư mục: {target_root} ---")
    
    if os.path.exists(target_root):
        print(f"Xóa thư mục cũ: {target_root}")
        shutil.rmtree(target_root)
    os.makedirs(os.path.join(target_root, "annotations"), exist_ok=True)

    # Map các split của Roboflow (train, valid, test)
    split_map = {
        'train': 'train',
        'valid': 'val',
        'test': 'test'
    }

    for rf_split, target_split in split_map.items():
        src_split_dir = os.path.join(download_dir, rf_split)
        
        if not os.path.exists(src_split_dir):
            continue
            
        print(f"Đang xử lý split: {rf_split} -> {target_split}")
        
        target_img_dir = os.path.join(target_root, target_split)
        os.makedirs(target_img_dir, exist_ok=True)

        # Di chuyển ảnh và file annotation
        files = os.listdir(src_split_dir)
        json_file = None
        
        for f in files:
            src_path = os.path.join(src_split_dir, f)
            
            if f.endswith(".json") or f == "_annotations.coco.json":
                json_file = f
            elif f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                shutil.move(src_path, os.path.join(target_img_dir, f))

        # Xử lý và chuyển đổi file annotation
        if json_file:
            src_json_path = os.path.join(src_split_dir, json_file)
            target_json_name = f"{target_split}.json"
            target_json_path = os.path.join(target_root, "annotations", target_json_name)
            
            shutil.move(src_json_path, target_json_path)
            
            # Chuyển đổi sang định dạng MOT
            target_mot_json_path = os.path.join(target_root, "annotations", f"{target_split}_mot.json")
            print(f"  Chuyển đổi annotation sang định dạng MOT...")
            convert_roboflow_to_mot(target_json_path, target_mot_json_path)
        else:
            print(f"  Cảnh báo: Không tìm thấy file JSON cho split {rf_split}")

    # Dọn dẹp thư mục tải về tạm thời
    try:
        shutil.rmtree(download_dir)
        print("Đã xóa thư mục tải về tạm thời.")
    except:
        pass

    print("\n--- HOÀN TẤT! ---")
    print(f"Dữ liệu đã sẵn sàng tại: {target_root}")

# Chạy hàm chính
if __name__ == "__main__":
    download_and_prepare()
