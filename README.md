# Network Intrusion Detection System using Deep Learning

**English** | [Tiếng Việt](#tổng-quan)

A deep learning-based Network Intrusion Detection System (NIDS) using the **CIC-IDS-2017** dataset. Multiple architectures are implemented and benchmarked: **DNN**, **1D-CNN**, and a deeper **3-block CNN**.

---

## Overview

- **Dataset:** [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) (8 CSV files, ~2.8M samples, 80+ features)
- **Task:** Binary classification (Benign vs Attack) + Multiclass classification (5 attack groups)
- **Models:**
  - DNN (3 hidden layers: 512 → 256 → 128)
  - 1D-CNN (2 Conv blocks + classifier)
  - CNN (3 Conv blocks: wide → medium → fine features)
- **Framework:** TensorFlow/Keras
- **Preprocessing:** MinMax scaling, undersampling, SMOTE, class weights

---

## Project Structure

```
├── src/                          # Reusable Python package
│   ├── config.py                 # Paths, constants, label mappings
│   ├── models/
│   │   ├── dnn.py                # DNN binary + multiclass builders
│   │   └── cnn_1d.py             # 1D-CNN / 3-block CNN builders
│   └── utils/
│       ├── callbacks.py          # Checkpoint + EarlyStopping + ReduceLR
│       ├── data_loader.py        # Load, reshape, scale helpers
│       ├── metrics.py            # Inference time, classification report
│       └── visualization.py      # Loss/acc curves, confusion matrix, ROC
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # Load, clean, encode, split, balance
│   ├── 02_dnn_binary.ipynb           # DNN — binary classification
│   ├── 03_dnn_multiclass.ipynb       # DNN — 15-class classification
│   ├── 04_dnn_multiclass_5.ipynb     # DNN — 5-group classification
│   ├── 05_cnn_multiclass.ipynb       # CNN (3-block) — 15-class
│   ├── 06_1dcnn_binary.ipynb         # 1D-CNN — binary classification
│   └── 07_1dcnn_multiclass.ipynb     # 1D-CNN — 5-group classification
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data Preprocessing Pipeline

1. **Concatenate** all 8 CSV files
2. **Drop duplicates** (reduces ~10% of data)
3. **Handle NaN/Infinity** — fill with median for skewed features, 0 otherwise
4. **Drop zero-variance columns** (dead features)
5. **Drop high-correlation columns** (> 0.95)
6. **Drop low-correlation columns** (< 0.01 with Label)
7. **Label encode** 15 classes → integers
8. **Train/Test split** 80/20 with stratification
9. **MinMax scaling**

### Binary preparation:
- Group all attack classes → `1`, keep Benign → `0`
- Undersample Benign to 500K samples
- Shuffle, compute class weights

### Multiclass preparation (5 groups):
- Map 15 original → 5 groups: `Benign`, `DoS/DDoS`, `PortScan`, `BruteForce`, `Other`
- Undersample Benign → 300K, SMOTE rare classes → 30K
- One-Hot encode labels

---

## Model Architectures

### DNN
```
Input (37) → Dense(512) → BN → Drop(0.2) → Dense(256) → BN → Drop(0.25)
          → Dense(128) → BN → Drop(0.25) → Output (Sigmoid / Softmax)
```

### 1D-CNN (2-block)
```
Input (37×1) → Conv1D(32, k=3) → BN → ReLU → MaxPool(2)
             → Conv1D(64, k=3) → BN → ReLU → MaxPool(2)
             → Flatten → Dense(128) → Drop(0.5) → Output
```

### CNN (3-block)
```
Input (37×1) → Conv1D(128, k=7) → BN → ReLU → MaxPool(2)
             → Conv1D(256, k=5) → BN → ReLU → MaxPool(2) → Drop(0.25)
             → Conv1D(512, k=3) → BN → ReLU → MaxPool(2) → Drop(0.3)
             → Flatten → Dense(256) → BN → Drop(0.4) → Output
```

---

## How to Run

### On Google Colab (recommended)
1. Upload the dataset to Google Drive
2. Open any notebook in `notebooks/` via Colab
3. Mount Drive and run all cells

### Locally
```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```
Adjust data paths in `src/config.py` to point to your local data directory.

---

## Results

Metrics are computed on the held-out test set (504,473 samples).

### Binary Classification — Scenario 1

| Model   | Accuracy | Precision (Attack) | Recall (Attack) | F1-Score (Attack) | AUC     | Parameters | Speed       |
|---------|----------|--------------------|-----------------|-------------------|---------|------------|-------------|
| DNN     | 98.41%   | 0.92               | 1.00            | 0.96              | 0.9993  | 187,393    | ~83 µs/pkt  |
| 1D-CNN  | 98.51%   | 0.92               | 1.00            | 0.96              | 0.9996  | 80,705     | ~92 µs/pkt  |

### Multiclass Classification (5 Groups) — Scenario 2

**DNN — Per-class metrics**

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Benign     | 1.00      | 0.97   | 0.99     |
| DoS/DDoS   | 0.97      | 1.00   | 0.98     |
| PortScan   | 0.74      | 1.00   | 0.85     |
| BruteForce | 0.87      | 0.89   | 0.88     |
| Other      | 0.10      | 0.99   | 0.18     |
| **Overall**| **—**     | **—**  | **—**    |
| Accuracy   | —         | —      | **97.33%** |

**1D-CNN — Per-class metrics**

| Class      | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Benign     | 1.00      | 0.98   | 0.99     |
| DoS/DDoS   | 0.98      | 1.00   | 0.99     |
| PortScan   | 0.74      | 1.00   | 0.85     |
| BruteForce | 0.94      | 0.88   | 0.91     |
| Other      | 0.23      | 1.00   | 0.37     |
| **Overall**| **—**     | **—**  | **—**    |
| Accuracy   | —         | —      | **98.10%** |

### Key Takeaways

| Criterion                | DNN               | 1D-CNN            | Winner      |
|--------------------------|-------------------|--------------------|-------------|
| Binary accuracy          | 98.41%            | 98.51%             | 1D-CNN      |
| Multiclass accuracy      | 97.33%            | 98.10%             | 1D-CNN      |
| Precision (Rare attacks) | 10.15%            | 22.94%             | 1D-CNN      |
| Recall (Rare attacks)    | ~99%              | ~99.6%             | 1D-CNN      |
| Training Stability       | Smooth            | Oscillating        | DNN         |
| Inference speed          | ~57.8 µs/sample   | ~86.7 µs/sample    | DNN         |
| Model size (params)      | 187,393           | 80,705             | 1D-CNN      |
| AUC (binary)             | 0.9993            | 0.9996             | 1D-CNN      |

---

## Full Report

📄 View the detailed results report:  
[`results/NIDS_Deep_Learning_Report_PhongNguyen.pdf`](results/NIDS_Deep_Learning_Report_PhongNguyen.pdf)

---

## License

This project is for educational and research purposes.

---

# Hệ thống Phát hiện Xâm nhập Mạng bằng Học Sâu

## Tổng quan

Hệ thống phát hiện xâm nhập mạng (NIDS) sử dụng học sâu trên bộ dữ liệu **CIC-IDS-2017**. Dự án triển khai và so sánh nhiều kiến trúc: **DNN**, **1D-CNN** và **CNN 3-block**.

- **Bộ dữ liệu:** [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) (8 file CSV, ~2.8M mẫu, 80+ đặc trưng)
- **Bài toán:** Nhị phân (Bình thường vs Tấn công) + Đa lớp (5 nhóm tấn công)
- **Framework:** TensorFlow/Keras
- **Tiền xử lý:** MinMax scaling, undersampling, SMOTE, trọng số lớp

## Pipeline tiền xử lý

1. **Nối** 8 file CSV
2. **Xóa hàng trùng lặp** (giảm ~10%)
3. **Xử lý NaN/Vô cực** — fill bằng trung vị (median) cho cột lệch, 0 cho còn lại
4. **Xóa cột chết** (phương sai = 0)
5. **Xóa cột tương quan cao** (> 0.95)
6. **Xóa cột tương quan thấp** (< 0.01 với nhãn)
7. **Mã hóa nhãn** 15 lớp → số nguyên
8. **Chia Train/Test** 80/20 (phân tầng)
9. **Chuẩn hóa MinMax**

### Nhị phân:
- Gộp tất cả tấn công → `1`, Benign → `0`
- Giảm Benign xuống 500K mẫu
- Trộn đều, tính trọng số lớp

### Đa lớp (5 nhóm):
- Gộp 15 nhãn gốc → 5 nhóm
- Giảm Benign → 300K, SMOTE lớp hiếm → 30K
- One-Hot encoding

## Cấu trúc thư mục

```
├── src/                          # Package Python tái sử dụng
│   ├── config.py                 # Đường dẫn, hằng số, mapping nhãn
│   ├── models/
│   │   ├── dnn.py                # Xây dựng DNN nhị phân / đa lớp
│   │   └── cnn_1d.py             # Xây dựng 1D-CNN / CNN 3-block
│   └── utils/
│       ├── callbacks.py          # Checkpoint + EarlyStopping + ReduceLR
│       ├── data_loader.py        # Load, reshape, scale dữ liệu
│       ├── metrics.py            # Đo tốc độ inference, báo cáo phân loại
│       └── visualization.py      # Biểu đồ loss/acc, confusion matrix, ROC
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # Load, làm sạch, mã hóa, chia, cân bằng
│   ├── 02_dnn_binary.ipynb           # DNN — phân loại nhị phân
│   ├── 03_dnn_multiclass.ipynb       # DNN — phân loại 15 lớp
│   ├── 04_dnn_multiclass_5.ipynb     # DNN — phân loại 5 nhóm
│   ├── 05_cnn_multiclass.ipynb       # CNN (3-block) — 15 lớp
│   ├── 06_1dcnn_binary.ipynb         # 1D-CNN — nhị phân
│   └── 07_1dcnn_multiclass.ipynb     # 1D-CNN — 5 nhóm
├── requirements.txt
├── .gitignore
└── README.md
```

## Cách chạy

### Trên Google Colab (khuyến nghị)
1. Upload bộ dữ liệu lên Google Drive
2. Mở notebook trong `notebooks/` qua Colab
3. Mount Drive và chạy toàn bộ cell

### Local
```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```
Chỉnh đường dẫn trong `src/config.py` cho phù hợp.

## Kiến trúc mô hình

| Mô hình | Task | Mô tả |
|---------|------|-------|
| DNN | Nhị phân | 3 lớp ẩn (512→256→128) + Dropout + BatchNorm |
| DNN | 15 lớp | Kiến trúc giống, softmax đầu ra |
| DNN | 5 nhóm | Giống, categorical crossentropy |
| 1D-CNN | Nhị phân | 2 block Conv1D(32→64) + Flatten + Sigmoid |
| 1D-CNN | 5 nhóm | 2 block Conv1D + Softmax |
| CNN | 15 lớp | 3 block Conv1D(128→256→512) + Softmax |

## Kết quả

Kết quả trên tập kiểm thử độc lập (504.473 mẫu).

### Phân loại Nhị phân — Kịch bản 1

| Mô hình | Accuracy | Precision (Tấn công) | Recall (Tấn công) | F1-Score (Tấn công) | AUC | Tham số | Tốc độ |
|---------|----------|----------------------|-------------------|---------------------|-----|---------|--------|
| DNN | 98,41% | 0,92 | 1,00 | 0,96 | 0,9993 | 187.393 | ~83 µs/gói |
| 1D-CNN | 98,51% | 0,92 | 1,00 | 0,96 | 0,9996 | 80.705 | ~92 µs/gói |

### Phân loại Đa lớp (5 nhóm) — Kịch bản 2

**DNN — Chỉ số từng nhóm**

| Nhóm | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Benign | 1,00 | 0,97 | 0,99 |
| DoS/DDoS | 0,97 | 1,00 | 0,98 |
| PortScan | 0,74 | 1,00 | 0,85 |
| BruteForce | 0,87 | 0,89 | 0,88 |
| Other | 0,10 | 0,99 | 0,18 |
| **Accuracy tổng** | | | **97,33%** |

**1D-CNN — Chỉ số từng nhóm**

| Nhóm | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Benign | 1,00 | 0,98 | 0,99 |
| DoS/DDoS | 0,98 | 1,00 | 0,99 |
| PortScan | 0,74 | 1,00 | 0,85 |
| BruteForce | 0,94 | 0,88 | 0,91 |
| Other | 0,23 | 1,00 | 0,37 |
| **Accuracy tổng** | | | **98,10%** |

### Tổng kết so sánh

| Tiêu chí | DNN | 1D-CNN | Kết luận |
|----------|-----|--------|----------|
| Accuracy nhị phân | 98,41% | 98,51% | 1D-CNN nhỉnh hơn |
| Accuracy đa lớp | 97,33% | 98,10% | 1D-CNN vượt trội |
| Precision nhóm Hiếm | 10,15% | 22,94% | 1D-CNN cải thiện gấp 2,3 lần |
| Recall nhóm Hiếm | ~99% | ~99,6% | Cả hai xuất sắc |
| Ổn định huấn luyện | Cao (mượt) | Thấp (dao động) | DNN ổn định hơn |
| Tốc độ suy luận | ~57,8 µs/mẫu | ~86,7 µs/mẫu | DNN nhanh hơn |
| Dung lượng mô hình | 187.393 params | 80.705 params | 1D-CNN nhẹ hơn |
| AUC (nhị phân) | 0,9993 | 0,9996 | 1D-CNN phân tách tốt hơn |

## Giấy phép

Dự án phục vụ mục đích giáo dục và nghiên cứu.

## Báo cáo kết quả

📄 Xem báo cáo chi tiết tại:  
[`results/NIDS_Deep_Learning_Report_PhongNguyen.pdf`](results/NIDS_Deep_Learning_Report_PhongNguyen.pdf)
