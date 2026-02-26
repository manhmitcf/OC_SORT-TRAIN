# Hướng dẫn chạy OC-SORT trên Kaggle (Phiên bản Git Clone)

Đây là hướng dẫn tối ưu để bạn chỉ cần push code lên Git, sau đó clone về Kaggle và chạy lệnh.

## 1. Chuẩn bị trên Kaggle
1.  Tạo Notebook mới.
2.  Settings: **GPU P100** (hoặc T4 x2), **Internet On**.

## 2. Các lệnh cần chạy (Copy vào từng cell)

### Cell 1: Clone Code & Cài đặt
```python
# 1. Clone repo của bạn (thay thế URL nếu bạn push lên repo riêng)
!git clone https://github.com/noahcao/OC_SORT.git
%cd OC_SORT

# 2. Cài đặt thư viện
!pip install loguru thop lap filterpy cython_bbox pandas xmltodict roboflow

# 3. === BƯỚC QUAN TRỌNG: BIÊN DỊCH C++ EXTENSION ===
# Lệnh này sẽ build module _C.so cần thiết cho việc evaluation
!python setup.py build_ext --inplace

# 4. Cài đặt dự án (quan trọng)
!pip install -e . --no-build-isolation
```

### Cell 2: Tải & Chuẩn bị Dữ liệu
Script `tools/prepare_data.py` đã được tích hợp sẵn để tải từ Roboflow và convert sang MOT format.

```python
# Chạy script chuẩn bị dữ liệu
!python tools/prepare_data.py
```

### Cell 3: Tải Weights & Train
File cấu hình `exps/example/mot/custom_finetune.py` đã được thiết lập sẵn cho Kaggle (Input size 800x1440).

```python
# 1. Cài đặt gdown để tải file từ Google Drive
!pip install gdown

# 2. Tải pre-trained weights từ link Google Drive (thêm --fuzzy để tải ổn định hơn)
!mkdir -p pretrained
!gdown --id 1RG8aCO5Kxi_XF_Rbfu58bvDcSdR6_2DU -O pretrained/ocsort_x_mot20.pth.tar --fuzzy

# 3. Chạy Train
!python tools/train.py -f exps/example/mot/custom_finetune.py -d 1 -b 8 --fp16 -o -c pretrained/ocsort_x_mot20.pth.tar
```

### Cell 4: Đánh giá trên tập Test
File cấu hình `exps/example/mot/custom_test.py` đã được tạo để chạy trên tập test. Script `tools/evaluate_custom.py` được thiết kế để tính toán và in ra các chỉ số HOTA, MOTA, IDF1.

```python
# Chạy đánh giá với model tốt nhất vừa train bằng script custom
!python tools/evaluate_custom.py -f exps/example/mot/custom_test.py -c YOLOX_outputs/custom_finetune/best_ckpt.pth.tar --fp16 --fuse
```

### Cell 5: Lưu kết quả
```python
!zip -r output.zip YOLOX_outputs/
from IPython.display import FileLink
FileLink(r'output.zip')
```
