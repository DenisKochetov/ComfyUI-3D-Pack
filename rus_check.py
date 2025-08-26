import os
import re

# ===== НАСТРОЙКИ =====
dir_to_check = r"C:\Users\user\ComfyUI\custom_nodes\ComfyUI-3D-Pack"   # папка для проверки
recursive = True                      # заходить ли в поддиректории (True/False)
limit = 50                             # сколько файлов максимум выводить
check_exts = {".py", ".md", ".txt"}   # какие расширения проверять (нижний регистр, с точкой)
# =====================

RUS_RE = re.compile(r"[А-Яа-яЁё]")

def iter_files(base_dir: str, recursive: bool):
    if recursive:
        for root, _, files in os.walk(base_dir):
            for f in files:
                yield os.path.join(root, f)
    else:
        for f in os.listdir(base_dir):
            p = os.path.join(base_dir, f)
            if os.path.isfile(p):
                yield p

def allowed_extension(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in check_exts

def first_russian_line(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for lineno, line in enumerate(f, start=1):
                if RUS_RE.search(line):
                    return lineno, line.rstrip("\n")
    except Exception:
        pass
    return None

found = 0
for path in iter_files(dir_to_check, recursive):
    if not allowed_extension(path):
        continue
    res = first_russian_line(path)
    if res:
        lineno, text = res
        print(path)
        print(f"{lineno}: {text}")
        print("-" * 80)
        found += 1
        if found >= limit:
            break
