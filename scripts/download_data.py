"""Tải ảnh của 3 bộ dữ liệu vào data/ (bỏ qua file đã có).

- Wicht V2: ảnh và nhãn .dat theo v2_train.desc, v2_test.desc
- Lexski, Murphy: ảnh theo trường "url" trong manifest.json

Chạy: python scripts/download_data.py
"""

import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WICHT_URL = "https://raw.githubusercontent.com/wichtounet/sudoku_dataset/master/"


def download(url, target):
    """Tải url về target nếu target chưa tồn tại."""
    if target.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        content = response.read()
    target.write_bytes(content)
    return True


def main():
    jobs = []
    # Wicht: mỗi ảnh đi kèm file nhãn .dat cùng tên
    for split in ("train", "test"):
        for line in (DATA / "wicht" / f"v2_{split}.desc").read_text(encoding="utf-8").split():
            for relative in (line, str(Path(line).with_suffix(".dat")).replace("\\", "/")):
                jobs.append((WICHT_URL + relative, DATA / "wicht" / relative))
    # Lexski, Murphy: đường dẫn nằm trong manifest
    for name in ("Lexski", "Murphy"):
        for row in json.loads((DATA / name / "manifest.json").read_text(encoding="utf-8")):
            jobs.append((row["url"], DATA / name / row["file"]))

    downloaded = failed = 0
    for url, target in jobs:
        try:
            downloaded += download(url, target)
        except OSError as error:
            failed += 1
            print(f"Lỗi tải {url}: {error}")
    print(f"Đã tải {downloaded} file mới, {len(jobs) - downloaded - failed} file đã có, {failed} lỗi.")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
