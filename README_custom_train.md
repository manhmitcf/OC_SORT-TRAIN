# Hướng dẫn Fine-tune OC-SORT với Custom Dataset (Roboflow)

Tài liệu này hướng dẫn cách fine-tune mô hình OC-SORT sử dụng pretrained weights `ocsort_x_mot20.pth.tar` với bộ dữ liệu custom (chỉ có label người) từ Roboflow.

## 1. Chuẩn bị dữ liệu

### 1.1. Tải và định dạng dữ liệu
1.  Tải bộ dữ liệu từ Roboflow dưới dạng **COCO format**.
2.  Giải nén và đặt vào thư mục `datasets` theo cấu trúc sau:

    ```
    OC_SORT/
    └── datasets/
        └── custom_dataset/
            ├── train/              # Chứa ảnh huấn luyện
            ├── val/                # Chứa ảnh validation (nếu có)
            └── annotations/
                ├── instances_train.json      # File annotation cho tập train (COCO format)
                └── instances_val.json        # File annotation cho tập val (COCO format)
    ```

### 1.2. Chuyển đổi định dạng JSON (Quan trọng)
Dữ liệu từ Roboflow thường thiếu các trường `video_id`, `frame_id` và `track_id` cần thiết cho OC-SORT. Ngoài ra, ID của class "person" có thể không phải là 1.

Sử dụng script `tools/convert_roboflow_to_mot.py` (đã được tạo sẵn) để chuẩn hóa file JSON. Chạy các lệnh sau từ thư mục gốc của dự án (OC_SORT):

```bash
# Chuyển đổi file train
python tools/convert_roboflow_to_mot.py datasets/custom_dataset/annotations/instances_train.json datasets/custom_dataset/annotations/train_mot.json

# Chuyển đổi file val
python tools/convert_roboflow_to_mot.py datasets/custom_dataset/annotations/instances_val.json datasets/custom_dataset/annotations/val_mot.json
```

**Lưu ý:** Sau khi chạy lệnh trên, hãy cập nhật file cấu hình (bước 2) để trỏ đến file mới (`train_mot.json` và `val_mot.json`).

## 2. Tạo file cấu hình (Exp File)

Tạo một file python mới tại `exps/example/mot/custom_finetune.py` với nội dung sau. File này kế thừa từ cấu hình của MOT20 nhưng trỏ đến dữ liệu của bạn.

```python
# exps/example/mot/custom_finetune.py
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
        
        # --- Cấu hình đường dẫn dữ liệu ---
        self.data_dir_name = "custom_dataset" # Tên thư mục trong datasets/
        # SỬ DỤNG FILE JSON ĐÃ ĐƯỢC CONVERT Ở BƯỚC 1.2
        self.train_ann = "annotations/train_mot.json" 
        self.val_ann = "annotations/val_mot.json"
        self.train_img_dir = "train"
        self.val_img_dir = "val" # Hoặc "train" nếu muốn validate trên tập train
        # ----------------------------------

        # Cấu hình input size (giữ nguyên theo MOT20 hoặc chỉnh lại tùy GPU)
        self.input_size = (896, 1600) 
        self.test_size = (896, 1600)
        self.random_size = (20, 36)
        
        # Cấu hình huấn luyện
        self.max_epoch = 20        # Số epoch fine-tune (ít hơn training from scratch)
        self.print_interval = 20
        self.eval_interval = 1
        self.test_conf = 0.001
        self.nmsthre = 0.7
        self.no_aug_epochs = 5
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        # Đường dẫn tuyệt đối đến thư mục dataset
        # Mặc định get_yolox_datadir() trả về <OC_SORT_ROOT>/datasets
        data_dir = os.path.join(get_yolox_datadir(), self.data_dir_name)

        dataset = MOTDataset(
            data_dir=data_dir,
            json_file=self.train_ann,
            name=self.train_img_dir, # Tên thư mục chứa ảnh train
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import MOTDataset, ValTransform

        data_dir = os.path.join(get_yolox_datadir(), self.data_dir_name)

        valdataset = MOTDataset(
            data_dir=data_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            name=self.val_img_dir, # Tên thư mục chứa ảnh val
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev=testdev)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
```

## 3. Lệnh Huấn Luyện (Fine-tune)

Sử dụng lệnh sau để bắt đầu fine-tune.

*   `-f`: Đường dẫn đến file cấu hình vừa tạo.
*   `-c`: Đường dẫn đến pretrained weights (`ocsort_x_mot20.pth.tar`).
*   `-b`: Batch size (giảm xuống nếu gặp lỗi Out of Memory, ví dụ: 8, 4).
*   `-d`: Số lượng GPU sử dụng (ví dụ: 1).

```shell
python tools/train.py -f exps/example/mot/custom_finetune.py -d 1 -b 8 --fp16 -o -c pretrained/ocsort_x_mot20.pth.tar
```

## 4. Lưu ý quan trọng

1.  **Pretrained Weights**: Đảm bảo file `ocsort_x_mot20.pth.tar` nằm đúng đường dẫn `pretrained/ocsort_x_mot20.pth.tar`. Nếu tên file khác, hãy sửa lại tham số `-c`.
2.  **Dataset Path**: Kiểm tra kỹ cấu trúc thư mục trong `datasets/custom_dataset`.
3.  **Classes**: Bộ dữ liệu Roboflow phải được export với class ID cho "person" khớp với mong đợi (thường là ID 1 trong COCO). Nếu model không học được gì, hãy kiểm tra lại `category_id` trong file json.
