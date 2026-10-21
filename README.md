# Sudoku Vision Solver

Bài tập lớn môn Trí tuệ nhân tạo: nhận diện và giải Sudoku từ ảnh.

Luồng: `Ảnh → tìm lưới → cắt 81 ô → nhận diện chữ số → giải backtracking`

## Cấu trúc

```
├── app.py                          # Giao diện Streamlit
├── requirements.txt
├── src/sudoku_ai/
│   ├── vision/grid.py              # NGƯỜI A: ảnh -> 81 ảnh ô
│   ├── recognition/digits.py       # NGƯỜI B: 81 ảnh ô -> bàn cờ 9x9
│   └── solver/backtracking.py      # NGƯỜI C: bàn cờ 9x9 -> lời giải
├── tests/
└── data/                           # Ảnh mẫu
```