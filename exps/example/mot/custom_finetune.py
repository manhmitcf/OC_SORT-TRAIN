# exps/example/mot/custom_finetune.py
import os
from yolox.exp import Exp as MyExp

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
