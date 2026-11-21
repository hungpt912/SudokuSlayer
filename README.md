# Sudoku Vision Solver

**Demo:** https://sudokuslayer.streamlit.app/

Bài tập lớn môn Trí tuệ nhân tạo: **nhận diện và giải Sudoku từ ảnh chụp**.

Hệ thống tìm lưới trong ảnh, đọc 81 ô bằng CNN rồi giải bằng thuật toán quay lui.

## Luồng xử lý

```text
Ảnh BGR
  -> Tiền xử lý: ảnh xám, lọc Gaussian                                (pipeline_image/preprocess.py)
  -> Tìm khung bảng, nắn phối cảnh về 450x450                         (pipeline_image/grid.py)
  -> Định vị đường kẻ, tách 81 ô, chuẩn hóa về 28x28                  (pipeline_image/cells.py)
  -> CNN hai đầu ra: ô có số không, nếu có là số mấy (1-9)?           (recognition/cnn_engine.py)
  -> Kiểm tra hợp lệ, giải bằng quay lui + bitmask + MRV              (solver/)
```

## Kết quả

Đánh giá trên 76 ảnh: 40 ảnh test Wicht V2, 30 ảnh Lexski, 6 ảnh Murphy. Chi tiết trong [`results/summary.md`](results/summary.md).

| Giai đoạn | Đúng/tổng | Tỉ lệ |
|---|---:|---:|
| Tìm lưới và tách ô (IoU khung >= 0.90) | 76/76 | 100.0% |
| Nhận diện đúng cả 81 ô | 63/76 | 82.9% |
| Giải đúng tự động từ ảnh | 62/76 | 81.6% |
| Bộ giải trên đề gốc | 75/76 | 98.7% |

- IoU = diện tích phần giao / diện tích phần hợp.
- Theo từng ô, CNN đọc đúng 99.19% (gồm ô trống).
- Ảnh duy nhất vô nghiệm (`Lexski/hnp3xzbl6k0d1.webp`) có một nước điền sai của người chơi.
- Thời gian xử lý trung bình khoảng 0.1–0.3 giây mỗi ảnh trên CPU, tùy máy.

## Cấu trúc thư mục

```text
├── requirements.txt            # Thư viện cần cài
├── pyproject.toml              # Khai báo gói sudoku_ai (chỉ cần khi huấn luyện)
├── pytest.ini                  # Cấu hình pytest
├── scripts/
│   └── download_data.py        # Tải ảnh của 3 bộ dữ liệu
├── src/sudoku_ai/
│   ├── config.py               # Hằng số dùng chung
│   ├── dataset.py              # Đọc danh sách ảnh, nhãn, góc bảng của Wicht V2
│   ├── pipeline_image/         # preprocess.py, grid.py, cells.py
│   ├── recognition/            # base.py, cnn_model.py, cnn_engine.py, train.py, witch_cnn.keras
│   └── solver/                 # validator.py, backtracking.py
├── tests/
│   ├── test_cnn.py             # Kiểm thử CNN (7 ca)
│   ├── test_backtracking.py    # Kiểm thử bộ giải (8 ca)
│   ├── run_tests.py            # Đánh giá toàn hệ thống trên 76 ảnh
│   └── _support.py             # Hàm hỗ trợ cho run_tests.py
├── results/                    # Kết quả đánh giá (tạo bởi run_tests.py)
└── data/                       # Nhãn và manifest (ảnh tải riêng)
```

## Chạy kiểm thử

Chạy các lệnh từ thư mục gốc dự án, theo đúng thứ tự dưới đây. Yêu cầu Python 3.10–3.13.

**Bước 1. Tạo môi trường và cài thư viện** (chỉ làm lần đầu)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

**Bước 2. Kiểm thử đơn vị** (không cần tải dữ liệu)

```bash
python -m pytest
```

Kết quả mong đợi: `15 passed`.

- `test_cnn.py` (7 ca) dùng mô hình `witch_cnn.keras` và ô trống tạo sẵn bằng NumPy.
- `test_backtracking.py` (8 ca) dùng hai đề Sudoku viết sẵn trong file test.

**Bước 3. Tải ảnh dữ liệu** (chỉ làm lần đầu, bỏ qua file đã có)

```bash
python scripts/download_data.py
```

Repo chỉ chứa nhãn và manifest; script tải ảnh về `data/`.

| Bộ dữ liệu | Vai trò | Nguồn |
|---|---|---|
| Wicht V2 | 160 ảnh train, 40 ảnh test | [wichtounet/sudoku_dataset](https://github.com/wichtounet/sudoku_dataset) (CC-BY-4.0) |
| Lexski | 30 ảnh test | [Lexski/sudoku-image-recognition](https://huggingface.co/datasets/Lexski/sudoku-image-recognition) |
| Murphy | 6 ảnh chụp màn hình test | [wmurphy41/sudoku_ocr](https://github.com/wmurphy41/sudoku_ocr) |

Tọa độ 4 góc bảng Wicht (`outlines_sorted.csv`) do Lars (@panexe) đóng góp.

**Bước 4. Đánh giá toàn hệ thống trên 76 ảnh**

```bash
python tests/run_tests.py                     # Cả 76 ảnh
python tests/run_tests.py --dataset murphy    # Chỉ một bộ: wicht | lexski | murphy
python tests/run_tests.py --limit 5           # Chạy thử 5 ảnh đầu
```

Mỗi ảnh chạy một lần, kết quả in ra màn hình. Mỗi lần chạy sẽ xóa kết quả cũ cùng loại.

- Chạy đủ 76 ảnh: ghi vào `results/`.
- Chạy một phần (`--dataset`, `--limit`): ghi vào `results/partial/` (không commit), không ghi đè kết quả đầy đủ.

**Bước 5. Xem kết quả** trong `results/`

| File | Nội dung |
|---|---|
| `summary.md` | Bảng tóm tắt độ chính xác và thời gian |
| `stage_summary.csv` | Độ chính xác và thời gian của từng giai đoạn |
| `per_image.csv` | Kết quả và thời gian từng ảnh |
| `config.json` | Môi trường đo, mã băm mô hình |
| `backtracking_reference/summary.json` | Bộ giải chạy riêng trên bảng nhãn đúng của dataset |
| `examples/` | Ảnh từng bước của 2 ảnh minh họa: `Wicht/image114.jpg` (đọc đúng), `Lexski/64580592jqzc1.webp` (đọc sai 4 thành 6) |
| `summary.json` | Số liệu chi tiết, chung và theo từng bộ dữ liệu |
| `audit.json`, `run.log`, `backtracking_reference/per_image.csv` | Nguồn nhãn, nhật ký, thời gian giải từng đề |
| `images/<bộ dữ liệu>/<ảnh>/` | Ảnh từng bước của cả 76 ảnh: `1_mask`, `2_detected_grid`, `3_warped_board`, `4_cnn_input`, `5_ocr_comparison.csv`, `6_solution` |


Cách tính:

- Xử lý ảnh được coi là đúng khi IoU khung ≥ 0.90 và đủ 81 ô hợp lệ.
- Nhận diện được coi là đúng khi cả 81 ô khớp đề gốc.
- Lời giải được coi là đúng khi thỏa luật Sudoku và giữ nguyên số gợi ý.
- Thời gian đo không gồm đọc/ghi đĩa, nạp mô hình và lượt chạy khởi động.

## Huấn luyện lại CNN

Cần làm xong Bước 1 và Bước 3 ở trên. Sau đó cài thêm gói `sudoku_ai` một lần:

```bash
pip install -e .
```

Lệnh này đọc `pyproject.toml` và đăng ký `src/sudoku_ai` thành gói Python, để chạy được `python -m sudoku_ai.recognition.train`. Kiểm thử ở Bước 2 và Bước 4 không cần lệnh này.

```bash
python -m sudoku_ai.recognition.train                                   # Từ đầu: 32 epoch, lr 7e-4
python -m sudoku_ai.recognition.train --initial-model src/sudoku_ai/recognition/witch_cnn.keras   # Tinh chỉnh: 5 epoch, lr 2e-4
```

- Dữ liệu huấn luyện: 160 ảnh train Wicht (cắt bằng chính pipeline, thêm bản làm mờ, tăng cường xoay/co giãn/dịch), tổng 51 840 ô, cộng 13 500 chữ số MNIST với trọng số 0.2.
- Mô hình mới ghi đè `witch_cnn.keras`; dùng `--output` để lưu ra file khác.

## Thành viên

| Thành viên | MSSV | Phụ trách |
|---|---|---|
| Nguyễn Quang Linh | 24020201 | Tiền xử lý ảnh, tìm lưới, tách ô |
| Phạm Thanh Hùng | 24020147 | Mô hình nhận diện chữ số (CNN) |
| Đinh Huyền Trang | 23020711 | Kiểm tra hợp lệ và thuật toán giải |