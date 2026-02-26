# exps/example/mot/custom_test.py
from .custom_finetune import Exp as CustomFinetuneExp

class Exp(CustomFinetuneExp):
    def __init__(self):
        super(Exp, self).__init__()
        # Ghi đè các thông số cho tập test
        self.val_ann = "test_mot.json"
        self.val_img_dir = "test"
        
        self.exp_name = "custom_test_evaluation"
