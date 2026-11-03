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
│       ├── recognition/            # Module nhận diện chữ số từ ảnh ô (OCR)
│       │   ├── __init__.py
│       │   ├── base.py             # Lớp nền tảng / Interface cho recognition
│       │   ├── cnn_engine.py     # Engine dự đoán bằng mô hình CNN
│       │   ├── cnn_model.py      # Định nghĩa kiến trúc mô hình CNN
│       │   ├── mnist_cnn.keras   # Trọng số mô hình CNN đã huấn luyện
│       │   ├── tesseract_engine.py # Engine nhận dạng bằng Tesseract OCR
│       │   └── train.py          # Script huấn luyện mô hình CNN
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
│   │   ├── backtracking.py
│   │   └── validator.py
│   ├── ui/                         # Test cho phần giao diện
│   │   └── board_view.py
│   ├── preview_pipeline_image.py   # Script xem trước kết quả pipeline xử lý ảnh
│   └── test_pipeline_image.py      # Test cho pipeline xử lý ảnh
│   ├── test_app_flow.py          # Kiểm thử luồng hoạt động chính
│   ├── test_cnn_accuracy.py      # Đánh giá độ chính xác của CNN
│   ├── test_cnn_engine.py        # Kiểm thử CNN engine
│   ├── test_cnn_model.py         # Kiểm thử kiến trúc mô hình CNN
│   └── test_predict_board.py     # Kiểm thử nhận dạng toàn bộ bàn cờ
│
└── data/                           # Dữ liệu dự án (ảnh, nhãn, model data)
    ├── mnist.npz                   # Dữ liệu cục bộ cho nhận diện
    ├── wicht/                      # Metadata và hướng dẫn
    └── wicht/images                # Dữ liệu ảnh Sudoku (tự tải và thêm vào)

Các module xử lý ảnh hiện là khung để bắt đầu hiện thực thuật toán.
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
| `data/wicht/images/` | **Không** (đã `.gitignore`) | Tự tải rồi giải nén vào đây, gồm các cặp `imageN.jpg` và `imageN.dat` (nhãn). Nguồn: [wichtounet/sudoku_dataset](https://github.com/wichtounet/sudoku_dataset), dùng bản V2 (`v2_training`, `v2_test`). |
| MNIST | Không cần đặt tay | `train.py` và `test_cnn_accuracy.py` gọi `tf.keras.datasets.mnist.load_data()`, Keras tự tải về `~/.keras/datasets/` lần đầu (cần mạng). |

Bộ Wicht dùng để thử pipeline xử lý ảnh (xem `dataset.py`), còn MNIST dùng để huấn luyện CNN nhận diện chữ số.

## Test pipeline_image, cells_image được phóng to lên thành 228x228 để phục vụ kiểm tra
python -m pytest tests/test_pipeline_image.py -v          # Pytest
python tests/preview_pipeline_image.py --limit 10         # Test 10 images
python tests/preview_pipeline_image.py --limit 0          # Test all images

## Nguồn dữ liệu và giấy phép

Bộ ảnh Sudoku của Baptiste Wicht và Jean Hennebert, phát hành theo giấy phép CC-BY-4.0. Nếu dùng trong báo cáo, trích dẫn:

> Wicht, B., Hennebert, J. "Camera-based Sudoku recognition with deep belief network." *6th International Conference of Soft Computing and Pattern Recognition (SoCPaR)*, 2014, pp. 83–88.

Tọa độ 4 góc bảng (`outlines_sorted.csv`) do Lars (@panexe) đóng góp. MNIST do Yann LeCun và cộng sự công bố.