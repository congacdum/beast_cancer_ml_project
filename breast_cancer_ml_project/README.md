# Breast Cancer Classification

Dự án **Machine Learning** dùng để phân loại mẫu ung thư vú thành hai nhóm
**Benign** hoặc **Malignant** dựa trên bộ dữ liệu Breast Cancer Wisconsin
Diagnostic.

Dự án cung cấp một quy trình đầy đủ cho bài toán phân loại nhị phân: đọc dữ
liệu, kiểm tra dữ liệu, tiền xử lý, huấn luyện mô hình, đánh giá mô hình, giải
thích kết quả, lưu artifacts, kiểm thử tự động và ứng dụng demo bằng
**Streamlit** cho dự đoán thủ công và dự đoán theo batch.

> Cảnh báo y tế: dự án chỉ phục vụ mục đích học tập và demo. Đây không phải là
> thiết bị y tế và không được dùng để thay thế tư vấn y khoa, xét nghiệm lâm
> sàng hoặc chẩn đoán của bác sĩ.

## Mục Lục

- [Tổng Quan Dự Án](#tổng-quan-dự-án)
- [Tính Năng Chính](#tính-năng-chính)
- [Dataset](#dataset)
- [Phương Pháp Xây Dựng Model](#phương-pháp-xây-dựng-model)
- [Kết Quả Đánh Giá](#kết-quả-đánh-giá)
- [Cài Đặt](#cài-đặt)
- [Cách Sử Dụng](#cách-sử-dụng)
- [Ứng Dụng Streamlit](#ứng-dụng-streamlit)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Artifacts Được Sinh Ra](#artifacts-được-sinh-ra)
- [Kiểm Thử](#kiểm-thử)
- [Thư Viện Yêu Cầu](#thư-viện-yêu-cầu)
- [Xử Lý Lỗi Thường Gặp](#xử-lý-lỗi-thường-gặp)
- [Hạn Chế](#hạn-chế)
- [Hướng Phát Triển](#hướng-phát-triển)

## Tổng Quan Dự Án

Dự án giải quyết bài toán **binary classification**:

| Lớp | Ý nghĩa | Nhãn mã hóa |
|---|---|---:|
| Benign | Mẫu u lành tính | 0 |
| Malignant | Mẫu u ác tính | 1 |

Mục tiêu kỹ thuật là xây dựng một pipeline rõ ràng, có thể chạy lại, dễ kiểm
tra và phù hợp để nộp bài hoặc demo. Mục tiêu Machine Learning là phát hiện
các ca **Malignant** với ưu tiên cao cho **Recall của lớp Malignant** và giảm
thiểu lỗi **False Negative**, vì bỏ sót một ca ác tính nguy hiểm hơn việc yêu
cầu kiểm tra thêm một ca lành tính.

## Tính Năng Chính

- Ứng dụng **Streamlit** gồm các trang giới thiệu, khám phá dữ liệu, huấn luyện,
  đánh giá và dự đoán.
- Pipeline huấn luyện có thể tái lập với `random_state=42` và
  `stratified train/test split`.
- Sáu mô hình Machine Learning cơ bản:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - SVM
  - KNN
  - Gaussian Naive Bayes
- Kiểm tra dữ liệu gồm schema, missing values, duplicate rows, invalid ranges,
  outliers, correlations và class balance.
- Quy tắc chọn best model ưu tiên tiêu chí y tế: Recall, FN, F1-score, ROC-AUC,
  Accuracy và khả năng giải thích.
- Có **cross-validation**, thử nghiệm **GridSearchCV** nhỏ và tối ưu
  **decision threshold**.
- Có artifacts giải thích mô hình bằng **SHAP** khi môi trường hỗ trợ, kèm
  fallback bằng feature importance.
- Hỗ trợ dự đoán thủ công và dự đoán theo batch.
- Hỗ trợ upload CSV/XLSX và xuất kết quả dự đoán ra Excel.
- Có bộ kiểm thử bằng `pytest` cho preprocessing, validation và prediction.

## Dataset

Dự án sử dụng bộ dữ liệu Breast Cancer Wisconsin Diagnostic được lưu cục bộ tại:

```text
data/breast_cancer_wisconsin.csv
```

Tóm tắt dataset theo artifacts hiện có:

| Thuộc tính | Giá trị |
|---|---:|
| Số mẫu | 569 |
| Số cột | 31 |
| Số feature | 30 |
| Target column | `diagnosis` |
| Số mẫu Benign | 357 |
| Số mẫu Malignant | 212 |
| Missing values trong file hiện tại | 0 |
| Duplicate rows trong file hiện tại | 0 |

Các feature là những chỉ số số học được trích xuất từ đặc điểm nhân tế bào,
ví dụ radius, texture, perimeter, area, smoothness, compactness, concavity,
concave points, symmetry và fractal dimension. Các nhóm feature có các biến
`mean`, `error` và `worst`.

## Phương Pháp Xây Dựng Model

### Tiền Xử Lý

Logic tiền xử lý nằm trong `src/preprocessing.py` và `src/data_validation.py`.

Các bước chính:

1. Tự động nhận diện target column từ `config.py` hoặc các tên ứng viên.
2. Chuẩn hóa missing tokens và nhãn dạng text.
3. Chuẩn hóa numeric strings nếu dữ liệu có chuỗi giống số.
4. Xóa các dòng thiếu target label.
5. Xóa duplicate rows trước khi chia tập dữ liệu để giảm nguy cơ
   train/test contamination.
6. Mã hóa target thành `Benign=0` và `Malignant=1`.
7. Chia dữ liệu bằng `train_test_split(..., stratify=y, random_state=42)`.
8. Xử lý missing values của feature bằng thống kê được fit trên training split.
9. One-hot encode và căn chỉnh feature columns giữa train/test.
10. Sinh báo cáo chất lượng dữ liệu và outlier flags.

### Huấn Luyện

Entrypoint huấn luyện là `src/train.py`.

```powershell
python src/train.py
```

hoặc:

```powershell
python -m src.train
```

Pipeline sẽ huấn luyện toàn bộ sáu model, đánh giá trên hold-out test set,
xếp hạng model theo quy tắc ưu tiên y tế, lưu artifacts và ghi báo cáo.

### Quy Tắc Chọn Model

Model cuối cùng được chọn theo thứ tự:

1. Recall của lớp Malignant cao nhất.
2. Số False Negative thấp nhất.
3. F1-score của lớp Malignant cao nhất.
4. ROC-AUC cao nhất.
5. Accuracy cao nhất.
6. Nếu vẫn bằng nhau, ưu tiên model đơn giản hơn hoặc dễ giải thích hơn.

Quy tắc này không dùng Accuracy làm tiêu chí duy nhất vì dự án được đặt trong
bối cảnh hỗ trợ sàng lọc y tế.

### Giải Thích Model

Pipeline sinh các artifacts giải thích:

- Global feature importance.
- SHAP summary plot nếu package `shap` khả dụng.
- Hình Decision Tree và file text rule.

Nếu SHAP không chạy được trong môi trường hiện tại, pipeline vẫn tiếp tục chạy
và dùng feature importance làm fallback.

## Kết Quả Đánh Giá

Bảng kết quả hiện tại chọn **Random Forest** là best model.

Metric trên hold-out test set của best model:

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.9737 |
| Precision (M) | 1.0000 |
| Recall (M) | 0.9286 |
| F1-score (M) | 0.9630 |
| Specificity (B) | 1.0000 |
| MCC | 0.9442 |
| ROC-AUC | 0.9954 |
| PR-AUC | 0.9929 |
| TN | 72 |
| FP | 0 |
| FN | 3 |
| TP | 39 |

Tóm tắt so sánh model:

| Model | Accuracy | Precision (M) | Recall (M) | F1-score (M) | ROC-AUC | FN |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9954 | 3 |
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9947 | 3 |
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9960 | 3 |
| KNN | 0.9561 | 0.9744 | 0.9048 | 0.9383 | 0.9825 | 4 |
| Naive Bayes | 0.9386 | 1.0000 | 0.8333 | 0.9091 | 0.9934 | 7 |
| Decision Tree | 0.9035 | 0.9189 | 0.8095 | 0.8608 | 0.9153 | 8 |

Các file báo cáo đầy đủ được sinh bởi training pipeline. Trong workspace hiện
tại, các kết quả đã sinh trước đó đang nằm trong `results/`. Theo `config.py`,
các kết quả mới khi chạy lại pipeline sẽ được ghi vào `reports/`.

## Cài Đặt

### 1. Tạo virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Nếu PowerShell chặn việc activate:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Cài đặt dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Kiểm tra môi trường

```powershell
python -m pytest
```

## Cách Sử Dụng

### Huấn luyện và đánh giá model

```powershell
python src/train.py
```

hoặc:

```powershell
python -m src.train
```

Sau khi chạy, pipeline sẽ tạo hoặc cập nhật model artifacts và reports theo
các đường dẫn được định nghĩa trong `config.py`.

### Dự đoán bằng Python

`src/predict.py` cung cấp hàm `predict_dataframe` để dùng trong code:

```python
import pandas as pd
from src.predict import load_model_assets, predict_dataframe

model, feature_names, label_mapping = load_model_assets()
input_df = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)
result = predict_dataframe(input_df, model, feature_names, label_mapping)
print(result)
```

### Dự đoán bằng CLI

`src/predict.py` cũng hỗ trợ input dạng JSON:

```powershell
python src/predict.py --json "{\"mean radius\": 14.0}"
```

JSON object phải chứa đầy đủ các feature được lưu trong
`models/feature_names.joblib`.

## Ứng Dụng Streamlit

Chạy app từ thư mục root của project:

```powershell
streamlit run app.py
```

hoặc:

```powershell
python -m streamlit run app.py
```

Các trang trong ứng dụng:

| Trang | Mục đích |
|---|---|
| Giới thiệu | Trình bày mục tiêu, bài toán phân loại nhị phân và tổng quan pipeline |
| Khám phá dữ liệu / EDA | Hiển thị dataset, shape, số feature, missing values, phân bố target, correlation heatmap và data quality report |
| Huấn luyện mô hình | Chạy full training pipeline, hiển thị best model gần nhất và artifacts đã lưu |
| Đánh giá mô hình | So sánh metrics, xem confusion matrix, ROC curves, threshold results, CV/tuning results và explainability plots |
| Dự đoán | Dự đoán thủ công, demo patients, batch upload, validation report và export Excel |

Batch prediction hỗ trợ:

- Upload CSV.
- Upload XLSX.
- Tải template từ `assets/breast_cancer_template.xlsx`.
- Validation trước khi predict.
- Export kết quả ra `prediction_results.xlsx`.

## Cấu Trúc Dự Án

```text
breast_cancer_ml_project/
|-- app.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- assets/
|   |-- breast_cancer_template.xlsx
|   `-- demo_patients.xlsx
|-- data/
|   `-- breast_cancer_wisconsin.csv
|-- models/
|   |-- best_model.joblib
|   |-- decision_threshold.joblib
|   |-- feature_names.joblib
|   |-- label_mapping.joblib
|   `-- <per-model>.joblib
|-- report/
|   `-- report.pdf
|-- results/
|   |-- model_comparison.csv
|   |-- model_comparison.md
|   |-- best_model_summary.md
|   |-- experiment_summary.md
|   |-- training_metadata.json
|   `-- figures/
|-- src/
|   |-- advanced_evaluation.py
|   |-- app_pages.py
|   |-- app_utils.py
|   |-- batch_prediction.py
|   |-- data_utils.py
|   |-- data_validation.py
|   |-- evaluate.py
|   |-- predict.py
|   |-- preprocessing.py
|   |-- train.py
|   |-- visualization.py
|   `-- __init__.py
`-- tests/
    |-- conftest.py
    |-- test_data_validation.py
    |-- test_model_prediction.py
    `-- test_preprocessing.py
```

Lưu ý quan trọng: `config.py` định nghĩa thư mục báo cáo sinh mới là
`reports/`. Nếu thư mục này chưa tồn tại, pipeline sẽ tự tạo khi chạy training.
Workspace hiện tại cũng có thư mục `results/` chứa các kết quả đã sinh trước đó.

## Artifacts Được Sinh Ra

### Model artifacts

| File | Mục đích |
|---|---|
| `models/best_model.joblib` | Model cuối cùng được dùng trong prediction workflow |
| `models/feature_names.joblib` | Schema feature theo đúng thứ tự lúc training |
| `models/label_mapping.joblib` | Mapping từ mã dự đoán sang nhãn |
| `models/decision_threshold.joblib` | Probability threshold dùng để dự đoán Malignant |
| `models/<model_name>.joblib` | Artifact riêng của từng baseline model |

### Report artifacts

Các report được sinh gồm:

- `model_comparison.csv`
- `model_comparison.md`
- `cross_validation_results.csv`
- `hyperparameter_tuning_results.csv`
- `threshold_optimization.csv`
- `best_model_summary.md`
- `experiment_summary.md`
- `training_metadata.json`
- `data_quality_report.json`
- `data_quality_report.html`
- `model_explainability.md`

### Figure artifacts

Các hình được sinh gồm:

- Target distribution.
- Correlation heatmap.
- Confusion matrix của best model.
- Confusion matrix của từng model.
- ROC curve của từng model.
- Combined ROC curve.
- Model metric comparison chart.
- Threshold optimization chart.
- Feature importance plot.
- SHAP summary plot.
- Decision Tree visualization.

## Kiểm Thử

Chạy toàn bộ test:

```powershell
python -m pytest
```

Các test hiện tại tập trung vào:

- Xử lý missing values.
- Xử lý duplicate rows.
- Chuẩn hóa label.
- Schema validation.
- Chuẩn hóa missing tokens.
- Chuẩn hóa numeric strings.
- Range validation.
- Load model artifacts.
- Cấu trúc output dự đoán.
- Kiểm tra probability nằm trong khoảng hợp lệ.

## Thư Viện Yêu Cầu

Các dependencies nằm trong `requirements.txt`:

| Package | Mục đích |
|---|---|
| `numpy` | Tính toán số học |
| `pandas` | Đọc dữ liệu, làm sạch dữ liệu và xử lý bảng |
| `scikit-learn` | Model ML, preprocessing, metrics, CV và tuning |
| `matplotlib` | Vẽ biểu đồ tĩnh |
| `seaborn` | Vẽ biểu đồ thống kê |
| `joblib` | Lưu và load model |
| `streamlit` | Web demo application |
| `shap` | Explainability plots |
| `openpyxl` | Đọc/ghi XLSX cho template, upload và export |
| `pytest` | Automated tests |

## Xử Lý Lỗi Thường Gặp

### `ModuleNotFoundError`

Cài lại dependencies từ project root:

```powershell
python -m pip install -r requirements.txt
```

Nên chạy các lệnh từ thư mục root của project để local imports hoạt động đúng.

### Streamlit không tìm thấy model

Hãy chạy training trước:

```powershell
python src/train.py
```

Sau đó khởi động lại Streamlit:

```powershell
python -m streamlit run app.py
```

### Thiếu reports

Chạy training pipeline. Các đường dẫn output được quản lý trong `config.py` và
sẽ được tạo tự động.

```powershell
python src/train.py
```

### PowerShell không activate được virtual environment

Dùng policy bypass trong phạm vi process hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\activate
```

### SHAP không khả dụng

Cài lại dependencies:

```powershell
python -m pip install -r requirements.txt
```

Nếu SHAP vẫn không chạy được trong môi trường hiện tại, training vẫn tiếp tục
và dùng feature importance làm fallback.

## Hạn Chế

- Dự án chỉ phục vụ học tập/demo, không phải hệ thống lâm sàng.
- Dataset nhỏ và là public benchmark; chưa có kiểm chứng trên dữ liệu bệnh viện
  độc lập.
- Hyperparameter tuning dùng search grid nhỏ để giữ thời gian chạy hợp lý.
- Cross-validation và tuning được dùng làm bằng chứng thực nghiệm; model cuối
  cùng vẫn được chọn theo rule hold-out ưu tiên y tế.
- UI Streamlit chưa có browser-level automated tests đầy đủ.
- SHAP phụ thuộc vào môi trường Python đang chạy.

## Hướng Phát Triển

- Kiểm chứng trên dataset y tế độc lập.
- Bổ sung nested cross-validation cho tuned models.
- Mở rộng hyperparameter search với giới hạn runtime rõ ràng.
- Thêm calibration analysis cho xác suất dự đoán.
- Thêm decision-curve analysis cho lựa chọn threshold trong bối cảnh y tế.
- Bổ sung automated integration tests cho Streamlit flows.
- Thêm experiment tracking bằng MLflow hoặc công cụ tương tự.
