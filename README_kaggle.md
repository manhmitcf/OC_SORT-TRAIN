# Hướng dẫn chạy OC-SORT trên Kaggle

Kaggle cung cấp môi trường Jupyter Notebook với GPU mạnh mẽ (P100 hoặc T4 x2), rất thích hợp để train model. Dưới đây là các bước chi tiết.

## 1. Chuẩn bị Notebook

1.  Tạo một Notebook mới trên Kaggle.
2.  Trong phần **Settings** (bên phải), chọn **Accelerator: GPU P100** (hoặc T4 x2).
3.  Bật **Internet: On**.

## 2. Clone Code và Cài đặt Môi trường

Copy đoạn code sau vào cell đầu tiên của Notebook để clone repo và cài đặt các thư viện cần thiết.

```python
# 1. Clone repo OC_SORT
!git clone https://github.com/noahcao/OC_SORT.git
%cd OC_SORT

# 2. Cài đặt các thư viện phụ thuộc
# Kaggle đã có sẵn PyTorch và CUDA, chỉ cần cài thêm các gói thiếu
!pip install -r requirements.txt
!pip install cython_bbox pandas xmltodict
!pip install roboflow

# 3. Cài đặt dự án ở chế độ develop
!python setup.py develop
```

## 3. Tải Dữ liệu từ Roboflow

Sử dụng script `download_custom_dataset.py` mà chúng ta đã tạo (bạn cần upload script này lên hoặc tạo mới ngay trong notebook).

Cách đơn giản nhất là tạo lại script tải dữ liệu ngay trong cell tiếp theo:

```python
# Tạo script tải dữ liệu
download_script = """
import os
import shutil
import sys
from roboflow import Roboflow

# Hàm convert (được nhúng trực tiếp để tiện chạy 1 file)
import json

def convert_roboflow_to_mot(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    person_cat_id = None
    for cat in data['categories']:
        if cat['name'] == 'person':
            person_cat_id = cat['id']
            break
    
    if person_cat_id is None:
        print(f"Error: 'person' category not found in {json_path}")
        return

    new_categories = [{"id": 1, "name": "person", "supercategory": "none"}]
    new_images = []
    
    for i, img in enumerate(data['images']):
        new_img = img.copy()
        if 'video_id' not in new_img: new_img['video_id'] = 1
        if 'frame_id' not in new_img:
            try:
                import re
                match = re.search(r'frame_(\d+)', new_img['file_name'])
                if match: new_img['frame_id'] = int(match.group(1))
                else: new_img['frame_id'] = i + 1
            except: new_img['frame_id'] = i + 1
        new_images.append(new_img)

    new_annotations = []
    for ann in data['annotations']:
        if ann['category_id'] == person_cat_id:
            new_ann = ann.copy()
            new_ann['category_id'] = 1
            if 'track_id' not in new_ann: new_ann['track_id'] = -1
            new_annotations.append(new_ann)

    new_data = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": new_categories,
        "images": new_images,
        "annotations": new_annotations
    }

    with open(output_path, 'w') as f:
        json.dump(new_data, f)
    print(f"Converted {json_path}")

# --- Main Download Logic ---
print("--- Downloading Dataset from Roboflow ---")
rf = Roboflow(api_key="7QSUEERZ8yV6mIjL8oiv")
project = rf.workspace("ok-vblps").project("dat_labeling")
version = project.version(8)
dataset = version.download("coco")

download_dir = dataset.location
target_root = "datasets/custom_dataset"

if os.path.exists(target_root): shutil.rmtree(target_root)
os.makedirs(target_root)
os.makedirs(os.path.join(target_root, "annotations"), exist_ok=True)

split_map = {'train': 'train', 'valid': 'val', 'test': 'test'}

for rf_split, target_split in split_map.items():
    src_split_dir = os.path.join(download_dir, rf_split)
    if not os.path.exists(src_split_dir): continue
        
    target_img_dir = os.path.join(target_root, target_split)
    os.makedirs(target_img_dir, exist_ok=True)

    files = os.listdir(src_split_dir)
    json_file = None
    for f in files:
        src_path = os.path.join(src_split_dir, f)
        if f.endswith(".json") or f == "_annotations.coco.json": json_file = f
        elif f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            shutil.move(src_path, os.path.join(target_img_dir, f))

    if json_file:
        src_json_path = os.path.join(src_split_dir, json_file)
        target_json_path = os.path.join(target_root, "annotations", f"{target_split}.json")
        shutil.move(src_json_path, target_json_path)
        
        # Convert
        target_mot_json_path = os.path.join(target_root, "annotations", f"{target_split}_mot.json")
        convert_roboflow_to_mot(target_json_path, target_mot_json_path)

try: shutil.rmtree(download_dir)
except: pass
print("Done!")
"""

with open("download_and_prep.py", "w") as f:
    f.write(download_script)

!python download_and_prep.py
```

## 4. Tải Pretrained Weights

```python
!mkdir -p pretrained
!wget -O pretrained/ocsort_x_mot20.pth.tar https://github.com/noahcao/OC_SORT/releases/download/v0.1/ocsort_x_mot20.pth.tar
# Lưu ý: Link trên là ví dụ, nếu không tải được, bạn cần upload file weights lên Kaggle Dataset và copy vào thư mục pretrained
```

## 5. Tạo File Cấu Hình (Exp File)

Tạo file `custom_finetune.py` trực tiếp trong notebook:

```python
exp_content = """
import os
import torch
import torch.distributed as dist
from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        self.data_dir_name = "custom_dataset"
        self.train_ann = "train_mot.json" 
        self.val_ann = "val_mot.json"
        self.train_img_dir = "train"
        self.val_img_dir = "val" 

        # Kaggle P100 có 16GB VRAM, có thể tăng input size lên lại
        self.input_size = (800, 1440) 
        self.test_size = (800, 1440)
        self.random_size = (18, 32)
        
        self.max_epoch = 20
        self.print_interval = 20
        self.eval_interval = 1
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.no_aug_epochs = 5
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import MOTDataset, TrainTransform, YoloBatchSampler, DataLoader, InfiniteSampler, MosaicDetection
        data_dir = os.path.join(get_yolox_datadir(), self.data_dir_name)
        dataset = MOTDataset(
            data_dir=data_dir, json_file=self.train_ann, name=self.train_img_dir, img_size=self.input_size,
            preproc=TrainTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_labels=500),
        )
        dataset = MosaicDetection(
            dataset, mosaic=not no_aug, img_size=self.input_size,
            preproc=TrainTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_labels=1000),
            degrees=self.degrees, translate=self.translate, scale=self.scale, shear=self.shear, perspective=self.perspective, enable_mixup=self.enable_mixup,
        )
        self.dataset = dataset
        if is_distributed: batch_size = batch_size // dist.get_world_size()
        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False, input_dimension=self.input_size, mosaic=not no_aug)
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "batch_sampler": batch_sampler}
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform
        data_dir = os.path.join(get_yolox_datadir(), self.data_dir_name)
        valdataset = MOTDataset(
            data_dir=data_dir, json_file=self.val_ann, img_size=self.test_size, name=self.val_img_dir,
            preproc=ValTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        )
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else: sampler = torch.utils.data.SequentialSampler(valdataset)
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "sampler": sampler, "batch_size": batch_size}
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(dataloader=val_loader, img_size=self.test_size, confthre=self.test_conf, nmsthre=self.nmsthre, num_classes=self.num_classes, testdev=testdev)
        return evaluator
"""

with open("exps/example/mot/custom_finetune.py", "w") as f:
    f.write(exp_content)
```

## 6. Chạy Train

```python
!python tools/train.py -f exps/example/mot/custom_finetune.py -d 1 -b 8 --fp16 -o -c pretrained/ocsort_x_mot20.pth.tar
```

## 7. Lưu Kết Quả

Sau khi train xong, kết quả sẽ nằm trong thư mục `YOLOX_outputs`. Bạn nên nén lại và tải về.

```python
!zip -r output.zip YOLOX_outputs/
from IPython.display import FileLink
FileLink(r'output.zip')
```
