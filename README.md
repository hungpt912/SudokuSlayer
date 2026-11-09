# Sudoku Vision Solver

Bài tập lớn môn Trí tuệ nhân tạo: Nhận diện và giải Sudoku từ ảnh.

Luồng xử lý: Đọc và tiền xử lý ảnh, tìm và nắn bảng, tách 81 ô, nhận diện chữ số, giải bằng backtracking.

## Cấu trúc dự án

```text
├── app.py                          # Entrypoint ứng dụng chính / Giao diện
├── requirement.txt                 # Các thư viện phụ thuộc[cite: 2]
├── README.md                       # Tài liệu hướng dẫn dự án
├── .gitignore                      # Cấu hình bỏ qua các file khi commit[cite: 1]
│
├── src/
│   └── sudoku_ai/                  # Package chính của dự án
│       ├── __init__.py
│       ├── config.py               # Kích thước và hằng số dùng chung
│       ├── dataset.py              # Đọc ảnh và nhãn tham chiếu Wicht V2
│       │
│       ├── pipeline_image/         # Xử lý ảnh và tách ô Sudoku
│       │
│       ├── recognition/            # Module nhận diện chữ số từ ảnh ô (OCR)
│       │   ├── __init__.py
│       │   ├── base.py             # Lớp nền tảng / Interface cho recognition
│       │   ├── cnn_engine.py       # Engine dự đoán bằng mô hình CNN
│       │   ├── cnn_model.py        # Định nghĩa kiến trúc mô hình CNN
│       │   ├── witch_cnn.keras     # Trọng số CNN huấn luyện từ Wicht V2 và MNIST
│       │   └── train.py            # Script huấn luyện mô hình CNN
│       │
│       ├── solver/                 # Module xử lý thuật toán giải Sudoku
│       │   ├── __init__.py
│       │   ├── backtracking.py     # Thuật toán giải Backtracking
│       │   └── validator.py        # Kiểm tra tính hợp lệ của bàn cờ
│       │
│       └── ui/                     # Module giao diện người dùng
│           └── board_view.py       # Hiển thị bàn cờ Sudoku
│
├── tests/                          # Các file test và script xem trước
│   ├── solver/                     # Unit test cho module solver
│   │   ├── test_backtracking.py
│   │   └── test_validator.py
│   ├── ui/                         # Test cho phần giao diện
│   │   └── test_board_view.py
│   ├── preview_pipeline_image.py   # Script xem trước kết quả pipeline xử lý ảnh
│   ├── benchmark_external_ocr.py  # Benchmark OCR trên Lexski và Murphy
│   ├── test_pipeline_image.py      # Test cho pipeline xử lý ảnh
│   ├── test_app_flow.py            # Kiểm thử luồng hoạt động chính
│   ├── test_cnn_accuracy.py        # Đánh giá OCR trên 40 ảnh Wicht V2 test
│   ├── test_cnn_engine.py          # Kiểm thử CNN engine
│   ├── test_cnn_model.py           # Kiểm thử kiến trúc mô hình CNN
│   ├── test_predict_board.py       # Kiểm thử nhận dạng toàn bộ bàn cờ
│   └── test_sudoku.py              # Chạy từ ảnh đầu vào đến lời giải
│
└── data/
    ├── Lexski/                     # Ảnh và nhãn kiểm thử Lexski
    ├── Murphy/                     # Ảnh và nhãn kiểm thử Murphy
    └── wicht/                      # Metadata và hướng dẫn
        └── images/                 # Dữ liệu ảnh Sudoku (tự tải và thêm vào)
```

## Phân công

- Người A phụ trách cả ba bước trong `pipeline_image/`: tiền xử lý, tìm/nắn bảng và tách/chuẩn hóa ô.
- Người B phụ trách nhận diện nội dung 81 ô và tạo ma trận Sudoku.
- Người C phụ trách kiểm tra bàn cờ và thuật toán giải Sudoku.

## Dữ liệu

Thư mục `data/` chứa:

| Thành phần | Có sẵn trong repo? | Ghi chú |
|---|---|---|
| `data/wicht/*.desc`, `outlines_sorted.csv`, `SOURCE_README.rst` | Có | Danh sách train/test (160 + 40 ảnh) và tọa độ 4 góc bảng của bộ Wicht V2. |
| `data/wicht/images/` | **Không** (đã `.gitignore`) | Tự tải rồi giải nén vào đây, gồm các cặp `imageN.jpg` và `imageN.dat` (nhãn). Nguồn: [wichtounet/sudoku_dataset](https://github.com/wichtounet/sudoku_dataset), dùng bản V2 (`v2_train`, `v2_test`). |
| `data/Lexski/`, `data/Murphy/` | Chỉ có file nhãn | Hai bộ ảnh kiểm thử bổ sung. |

Wicht V2 dùng 160 ảnh để huấn luyện và 40 ảnh để kiểm thử theo danh sách chính thức.
Huấn luyện bổ sung 13.500 mẫu MNIST từ số `1` đến `9` với trọng số `0,2` mỗi mẫu, so với `1,0` cho Wicht.

## Nguồn dữ liệu và giấy phép

Bộ ảnh Sudoku của Baptiste Wicht và Jean Hennebert, phát hành theo giấy phép CC-BY-4.0. 

> Wicht, B., Hennebert, J. "Camera-based Sudoku recognition with deep belief network." *6th International Conference of Soft Computing and Pattern Recognition (SoCPaR)*, 2014, pp. 83–88.

Tọa độ 4 góc bảng (`outlines_sorted.csv`) do Lars (@panexe) đóng góp.
