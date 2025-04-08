import os
import shutil


def find_file(path, filename):
    for root, dirs, files in os.walk(path):
        if filename in files and "torch/nn/modules" in root:
            return os.path.join(root, filename)
    return None


target_path = find_file("/", "pooling.py")
print(f"Found target path: {target_path}")

if target_path:
    shutil.copy("/root/pooling.py", target_path)
    print(f"Copied pooling.py to {target_path}")
else:
    print("Warning: Target path not found.")
    raise ValueError
