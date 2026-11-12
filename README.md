# Flower Classification

Ứng dụng Flask phân loại ảnh hoa thành 4 nhóm: daisy, rose, sunflower và tulip. Tải ảnh lên để xem dự đoán của Random Forest và CNN.

## Dataset
Link lưu Dataset: [Dataset](https://drive.google.com/drive/folders/1Fv6cm6c_nDUfvYl8SdBlEQTEDd8ELDxL?usp=sharing).

Dùng khi muốn chạy lại notebook hoặc huấn luyện model.

## Model
Đảm bảo thư mục `flower_classification/model/` có đủ:

```text
rf_model.joblib
scaler.joblib
label_encoder.joblib
cnn_model.h5
```

Lưu ý: `cnn_model.h5` chưa được commit lên Git vì dung lượng lớn. Cần đặt file này vào đúng thư mục `model/` trước khi chạy ứng dụng. Tải model tại link: [Model CNN](https://drive.google.com/file/d/174lxmoZhdocHpGoNvm-wC2KzJgTRSoDQ/view?usp=sharing).

## Chạy ứng dụng
Sau khi clone repository về máy, mở terminal tại thư mục gốc CS231 và chạy lần lượt các lệnh:
```bash
python -m venv .venv
source .venv/bin/activate        # Với Windows thì gõ: .venv\Scripts\activate
pip install -r requirements.txt
cd flower_classification
python app.py
```
Trong requirements.txt bao gồm các thư viện cần thiết cho cả ứng dụng và notebook. 

Mở `http://127.0.0.1:5000` nếu trình duyệt không tự bật. Chọn ảnh hoa và nhấn **Predict** để xem kết quả.