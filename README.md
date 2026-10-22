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
│   ├── pipeline_image/             # NGƯỜI A: ảnh Sudoku thành 81 ảnh ô
│   │   ├── __init__.py
│   │   ├── preprocess.py           # Đọc ảnh và tiền xử lý
│   │   ├── grid.py                 # Tìm bảng và nắn phối cảnh
│   │   └── cells.py                # Tách ô và chuẩn hóa
│   ├── recognition/                # NGƯỜI B: ảnh ô thành bàn cờ 9×9
│   ├── solver/                     # NGƯỜI C: kiểm tra và giải bàn cờ
│   └── ui/                         # Khung hiển thị, thực hiện sau
├── tests/
└── data/                           # Dữ liệu dùng chung; hiện có mnist.npz
```

Các module xử lý ảnh hiện là khung để bắt đầu hiện thực thuật toán.

## Phân công

- Người A phụ trách cả ba bước trong `pipeline_image/`: tiền xử lý, tìm/nắn bảng và tách/chuẩn hóa ô.
- Người B phụ trách nhận diện nội dung 81 ô và tạo ma trận Sudoku.
- Người C phụ trách kiểm tra bàn cờ và thuật toán giải Sudoku.