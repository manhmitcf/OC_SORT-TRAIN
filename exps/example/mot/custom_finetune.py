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
        self.exp_name = "custom_finetune"
        
        # Đường dẫn đến dữ liệu đã convert
        self.data_dir_name = "custom_dataset"
        self.train_ann = "train_mot.json" 
        self.val_ann = "val_mot.json"
        self.train_img_dir = "train"
        self.val_img_dir = "val" 

        # Cấu hình cho GPU Kaggle (16GB VRAM)
        self.input_size = (800, 1440) 
        self.test_size = (800, 1440)
        self.random_size = (18, 32)
        
        self.max_epoch = 20
        self.eval_interval = 1 # Đánh giá trên tập val sau mỗi epoch
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

    # ==============================================================================
    # GHI ĐÈ CÁC HÀM LOADER ĐỂ TRÁNH PHỤ THUỘC VÀO CÁC FILE BỊ THIẾU
    # ==============================================================================
    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        data_dir = os.path.join(get_yolox_datadir(), self.data_dir_name)

        dataset = MOTDataset(
            data_dir=data_dir,
            json_file=self.train_ann,
            name=self.train_img_dir,
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

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

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
            name=self.val_img_dir,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
        }
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
