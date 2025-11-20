# utils/path.py
import sys
from pathlib import Path

def add_root_to_path():
    root = Path(__file__).parent.parent.resolve()  # từ utils → lên 2 cấp
    if str(root) not in sys.path:
        sys.path.append(str(root))
    return root

# Gọi ở bất kỳ file nào:
# from utils.path import add_root_to_path
# add_root_to_path()