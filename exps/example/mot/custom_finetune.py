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
        # SỬ DỤNG FILE JSON ĐÃ ĐƯỢC CONVERT
        self.train_ann = "train_mot.json" 
        self.val_ann = "val_mot.json"
        self.train_img_dir = "train"
        self.val_img_dir = "val" 
        # ----------------------------------

        # Cấu hình input size (giảm xuống để tránh OOM trên GPU 4GB)
        # Gốc: (896, 1600) -> Giảm xuống: (608, 1088) hoặc (512, 928)
        self.input_size = (896, 1600)
        self.test_size = (896, 1600)
        self.random_size = (14, 26) # Giảm range random size tương ứng
        
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
