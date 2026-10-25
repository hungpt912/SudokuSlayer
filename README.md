# Sudoku Vision Solver

Bài tập lớn môn Trí tuệ nhân tạo: nhận diện và giải Sudoku từ ảnh.

Luồng xử lý: đọc và tiền xử lý ảnh, tìm và nắn bảng, tách 81 ô, nhận diện chữ số, giải bằng backtracking.

## Cấu trúc

```text
├── app.py                          # Khung giao diện, thực hiện sau
├── requirement.txt
├── src/sudoku_ai/
│   ├── __init__.py
│   ├── config.py                   # Kích thước và hằng số dùng chung
│   ├── dataset.py                  # Đọc ảnh và nhãn tham chiếu Wicht V2
│   ├── pipeline_image/             # NGƯỜI A: ảnh Sudoku thành 81 ảnh ô
│   │   ├── __init__.py
│   │   ├── preprocess.py           # Đọc ảnh và tiền xử lý
│   │   ├── grid.py                 # Tìm bảng và nắn phối cảnh
│   │   └── cells.py                # Tách ô và chuẩn hóa
│   ├── recognition/                # NGƯỜI B: ảnh ô thành bàn cờ 9×9
│   ├── solver/                     # NGƯỜI C: kiểm tra và giải bàn cờ
│   └── ui/                         # Khung hiển thị, thực hiện sau
├── tests          
└── data/
    ├── mnist.npz                   # Dữ liệu cục bộ cho nhận diện
    ├── wicht/                      # Metadata và hướng dẫn
    └── wicht/images                # Chứa các file image.dat và image.jpg, dùng để làm dữ liệu ảnh chạy Sudoku, tự tải và thêm vào để sử dụng
```

Các module xử lý ảnh hiện là khung để bắt đầu hiện thực thuật toán.

## Phân công

- Người A phụ trách cả ba bước trong `pipeline_image/`: tiền xử lý, tìm/nắn bảng và tách/chuẩn hóa ô.
- Người B phụ trách nhận diện nội dung 81 ô và tạo ma trận Sudoku.
- Người C phụ trách kiểm tra bàn cờ và thuật toán giải Sudoku.

## Test pipeline_image, cells_image được phóng to lên thành 228x228 để phục vụ kiểm tra
python -m pytest tests/test_pipeline_image.py -v          # Pytest
python tests/preview_pipeline_image.py --limit 10         # Test 10 images
python tests/preview_pipeline_image.py --limit 0          # Test all images
