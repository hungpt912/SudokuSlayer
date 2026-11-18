# Kết quả đánh giá toàn hệ thống

Số ảnh: 76.

| Giai đoạn | Đúng/tổng | Độ chính xác (%) | Thời gian TB (ms) |
|---|---:|---:|---:|
| Xử lý ảnh (tìm lưới, tách ô) | 76/76 | 100.00 | 152.289 |
| Nhận diện CNN (đúng cả 81 ô) | 63/76 | 82.89 | 60.428 |
| Giải trên kết quả OCR | 62/76 | 81.58 | 4.171 |
| Toàn hệ thống | 62/76 | 81.58 | 216.907 |

CNN theo từng ô (gồm ô trống): 6106/6156 = 99.19%.

| Bộ dữ liệu | Số ảnh | OCR đúng cả bảng | Giải đúng tự động |
|---|---:|---:|---:|
| Lexski | 30 | 18/30 | 17/30 |
| Murphy | 6 | 6/6 | 6/6 |
| Wicht | 40 | 39/40 | 39/40 |

| Bước | TB (ms) | Trung vị (ms) |
|---|---:|---:|
| decode_ms | 9.386 | 3.746 |
| preprocess_ms | 10.407 | 6.511 |
| grid_ms | 60.896 | 42.642 |
| cells_ms | 71.581 | 80.859 |
| ocr_ms | 60.428 | 61.796 |
| backtracking_ms | 4.171 | 1.478 |
| total_ms | 216.907 | 199.625 |
| pipeline_ms | 152.289 | 134.552 |

Bộ giải trên đề gốc: 75/76 bảng có lời giải; TB 2.197 ms, trung vị 0.476 ms.
Bảng vô nghiệm: Lexski/hnp3xzbl6k0d1.webp

Ghi chú: thời gian không gồm đọc/ghi đĩa, nạp mô hình và chạy khởi động. Xử lý ảnh đúng khi IoU khung >= 0.90 và đủ 81 ô hợp lệ. Lời giải đúng khi thỏa luật Sudoku và giữ nguyên số gợi ý gốc.
