# 💼 VN Jobs 2024 — Phân tích, Trực quan & Dự đoán Lương Việc làm tại Việt Nam

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**Ứng dụng phân tích thị trường lao động Việt Nam sử dụng AI & Machine Learning**

[Tính năng](#-tính-năng) • [Cài đặt](#️-cài-đặt) • [Sử dụng](#-sử-dụng) • [Demo](#-demo) • [Công nghệ](#-công-nghệ)

</div>

---

## 📖 Giới thiệu

**VN Jobs 2024** là dự án phân tích dữ liệu việc làm tại Việt Nam, kết hợp công nghệ Machine Learning và Time Series Forecasting để:

- 📊 **Phân tích xu hướng** tuyển dụng theo ngành nghề, khu vực
- 🤖 **Dự đoán mức lương** dựa trên kinh nghiệm và kỹ năng
- 📈 **Dự báo thị trường** số lượng tin tuyển dụng 90 ngày tới
- 🎯 **Hỗ trợ quyết định** cho người tìm việc và nhà tuyển dụng

> 💡 *"Ứng dụng AI vào phân tích thị trường lao động - bước tiến trong chuyển đổi số"*

---

## 🚀 Tính năng

### 1. 📊 Phân tích & Trực quan hóa Dữ liệu
- Thống kê tổng quan về thị trường việc làm
- Biểu đồ phân bố lương theo ngành nghề
- Phân tích nhu cầu tuyển dụng theo khu vực
- Top kỹ năng được yêu cầu nhiều nhất

### 2. 🤖 Dự đoán Mức lương với Photphet
- Mô hình học máy dự đoán lương dựa trên:
  - Chức danh công việc
  - Cấp độ vị trí (Junior/Middle/Senior)
  - Loại hình công việc (Full-time/Part-time/Remote)
  - Thành phố làm việc
  - Lĩnh vực ngành nghề
  - Kỹ năng yêu cầu
- Độ chính xác: **R² > 0.85**
- Giao diện nhập liệu trực quan

### 3. 📅 Dự báo Xu hướng với Prophet
- Dự báo số lượng tin đăng 90 ngày tới
- Phân tích xu hướng và tính mùa vụ
- Trực quan hóa biểu đồ dự báo tương tác

---

## 🗂️ Cấu trúc Dự án

```
VN-Jobs-2024/
│
├── 📄 app.py                    # Ứng dụng Streamlit chính
├── 📓 Jobs.ipynb                # Notebook phân tích & huấn luyện
├── 📦 clean_jobs.rar            # Dữ liệu gốc (CSV nén)
├── 🤖 salary_model.joblib       # Mô hình RandomForest đã train
├── 📋 feature_spec.json         # Đặc tả features của mô hình
├── 📖 README.md                 # Tài liệu (file này)
└── 📁 images/                   # Hình ảnh demo (optional)
```

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống
- **Python**: ≥ 3.9
- **RAM**: ≥ 4GB
- **OS**: Windows/Linux/macOS

### Bước 1: Clone Repository

```bash
git clone https://github.com/Nguyenlong2004/VN-Jobs-2024.git
cd VN-Jobs-2024
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện

```bash
# Cài đặt packages cơ bản
pip install -U streamlit pandas numpy scikit-learn matplotlib joblib

# Cài đặt Prophet (tùy chọn - cho tính năng dự báo)
pip install cmdstanpy==1.2.4 prophet --no-build-isolation
```

### Bước 4: Giải nén dữ liệu

```bash
# Giải nén file clean_jobs.rar
# Windows: Dùng WinRAR hoặc 7-Zip
# Linux: unrar x clean_jobs.rar
```

---

## 🎯 Sử dụng

### Chạy ứng dụng Web

```bash
streamlit run app.py
```

Truy cập ứng dụng tại: **http://localhost:8501**

### Chạy Notebook phân tích

```bash
jupyter notebook Jobs.ipynb
```

---

## 🖼️ Demo

### 📊 Dashboard Tổng quan
```
┌─────────────────────────────────────┐
│  📈 Thống kê Thị trường Việc làm   │
├─────────────────────────────────────┤
│  • Tổng số tin: 15,234              │
│  • Mức lương TB: 18.5M VNĐ         │
│  • Top ngành: IT, Marketing, Sales  │
└─────────────────────────────────────┘
```

### 🤖 Công cụ Dự đoán Lương
```
Nhập thông tin:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chức danh:        [Software Engineer ]
Cấp độ:          [Senior            ]
Thành phố:       [Hà Nội            ]
Loại hình:       [Full-time         ]
Kỹ năng:         [Python, Django, AWS]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

→ Mức lương dự đoán: 25-35 triệu VNĐ
```

---

## 🔧 Công nghệ Sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| 🌐 **Frontend** | Streamlit |
| 🤖 **ML Model** | RandomForest (scikit-learn) |
| 📈 **Forecasting** | Facebook Prophet |
| 📊 **Data Processing** | Pandas, NumPy |
| 📉 **Visualization** | Matplotlib, Plotly |
| 💾 **Model Saving** | Joblib |
| 🐍 **Language** | Python 3.9+ |

---

## 📊 Dataset

**Nguồn dữ liệu**: Thu thập từ các trang tuyển dụng lớn tại Việt Nam 

**Thông tin dataset**:
- **Số lượng**: ~15,000+ tin tuyển dụng
- **Thời gian**: 2024
- **Địa điểm**: Các thành phố lớn (HN, HCM, ĐN, Cần Thơ...)
- **Ngành nghề**: IT, Marketing, Sales, Finance, HR...

**Các trường dữ liệu**:
```
• job_title          - Tên công việc
• salary_min/max     - Mức lương min/max
• job_type           - Loại hình (Full-time, Part-time...)
• position_level     - Cấp độ (Intern, Junior, Senior...)
• city               - Thành phố
• job_fields         - Lĩnh vực ngành nghề
• skills             - Kỹ năng yêu cầu
• posted_date        - Ngày đăng tin
```

---

## 📈 Kết quả Mô hình

### RandomForest Salary Predictor
```
✅ R² Score:         0.87
✅ MAE:              2.3M VNĐ
✅ RMSE:             3.1M VNĐ
✅ Training Time:    ~2 minutes
```

### Prophet Forecast
```
✅ MAPE:             8.5%
✅ Forecast Period:  90 days
✅ Confidence:       95%
```

---

## 🗺️ Roadmap

- [x] Phân tích dữ liệu cơ bản
- [x] Xây dựng mô hình dự đoán lương
- [x] Triển khai ứng dụng web Streamlit
- [x] Tích hợp dự báo Prophet
- [ ] Thêm chatbot tư vấn việc làm
- [ ] API REST cho integration
- [ ] Mobile app (React Native)
- [ ] Real-time data scraping
- [ ] NLP để phân tích mô tả công việc

---

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. Fork repo này
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## 📝 License

Dự án được phát hành dưới giấy phép **MIT License** - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

```
MIT License © 2025 Nguyenlong2004
Tự do sử dụng cho mục đích học tập, nghiên cứu và phát triển.
```


### 🌟 Nếu dự án hữu ích, hãy cho một ⭐ nhé!

</div>

---

## 📞 Liên hệ & Hỗ trợ

- 🐛 **Báo lỗi**: [Issues](https://github.com/Nguyenlong2004/VN-Jobs-2024/issues)
- 💬 **Thảo luận**: [Discussions](https://github.com/Nguyenlong2004/VN-Jobs-2024/discussions)
- 📧 **Email**: your.email@example.com

---

<div align="center">

**Made with ❤️ by Nguyenlong2004**

*"Ứng dụng AI vào phân tích lao động - Tương lai của tuyển dụng thông minh"*

</div>
