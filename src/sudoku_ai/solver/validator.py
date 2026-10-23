"""Kiểm tra tính hợp lệ của bàn cờ Sudoku."""

def is_valid_placement(grid, row, col, num):
    """Đặt `num` vào grid[row][col] có hợp lệ không (không trùng hàng, cột, khối 3x3)."""
    if num in grid[row]:
        return False

    if num in [grid[r][col] for r in range(9)]:
        return False

    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] == num:
                return False

    return True


def is_valid_board(grid):
    """Các số đã điền có mâu thuẫn nhau không (dùng để phát hiện nhận diện sai)."""
    for row in range(9):
        for col in range(9):
            num = grid[row][col]
            if num == 0:
                continue
            grid[row][col] = 0                              # bỏ tạm chính nó ra
            ok = is_valid_placement(grid, row, col, num)
            grid[row][col] = num                            # trả lại
            if not ok:
                return False
    return True