import numpy as np
import json
import os

base_dir = r"D:\ARFI\Lomba\datathon-ui\Datathon-UI\Dataset"

def check_npy(path):
    try:
        arr = np.load(path)
        print(f"[{os.path.basename(path)}] Shape: {arr.shape}, dtype: {arr.dtype}")
    except Exception as e:
        print(f"[{os.path.basename(path)}] Error: {e}")

def check_json(path, n_samples=1):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"[{os.path.basename(path)}] Type: {type(data)}, Length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
            if isinstance(data, list) and len(data) > 0:
                print(f"Sample: {str(data[:n_samples])[:200]}...")
            elif isinstance(data, dict):
                keys = list(data.keys())
                print(f"Keys (first 5): {keys[:5]}")
                print(f"Sample value: {str(data[keys[0]])[:200]}...")
    except Exception as e:
        print(f"[{os.path.basename(path)}] Error: {e}")

print("--- STATIC ---")
check_npy(os.path.join(base_dir, "static", "matrix.npy"))
check_npy(os.path.join(base_dir, "static", "active_mask.npy"))
check_json(os.path.join(base_dir, "static", "Roads1260.json"))

print("\n--- TRAIN ---")
check_npy(os.path.join(base_dir, "train", "train_speed_m1_1_11160.npy"))
check_json(os.path.join(base_dir, "train", "train_text_m1_1_11160.json"))

print("\n--- TEST ---")
check_npy(os.path.join(base_dir, "test", "test_X_hist.npy"))
check_json(os.path.join(base_dir, "test", "test_texts.json"))
