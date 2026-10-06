"""
Datathon Task 1: City Network Traffic Forecasting
Baseline Pipeline - LightGBM
All-in-one script: preprocessing, feature engineering, training, inference, submission.
"""

import numpy as np
import json
import lightgbm as lgb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os
import gc
import time
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "Dataset"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_ROADS = 1260
HIST_LEN = 15          # 15 timesteps of history
HORIZONS = [5, 10, 15] # predict +5, +10, +15 steps ahead (=+20, +40, +60 min)
N_TEXT_FEATURES = 10   # TF-IDF -> SVD components
N_FOLDS = 3            # number of CV folds (keep small for speed)
SEED = 42

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 1000,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# ============================================================
# 1. LOAD DATA
# ============================================================
def load_data():
    print("=" * 60)
    print("[1] Loading data...")
    t0 = time.time()
    
    # Speed data (train)
    speed_m1 = np.load(os.path.join(DATA_DIR, "train", "train_speed_m1_1_11160.npy"))  # (11160, 1260)
    speed_m2 = np.load(os.path.join(DATA_DIR, "train", "train_speed_m2_1_5039.npy"))   # (5039, 1260)
    
    # Text data (train)
    text_m1 = json.load(open(os.path.join(DATA_DIR, "train", "train_text_m1_1_11160.json"), encoding="utf-8"))
    text_m2 = json.load(open(os.path.join(DATA_DIR, "train", "train_text_m2_1_5039.json"), encoding="utf-8"))
    
    # Test data
    test_X_hist = np.load(os.path.join(DATA_DIR, "test", "test_X_hist.npy"))  # (540, 15, 1260)
    test_texts = json.load(open(os.path.join(DATA_DIR, "test", "test_texts.json"), encoding="utf-8"))
    
    # Adjacency matrix
    adj = np.load(os.path.join(DATA_DIR, "static", "matrix.npy"))  # (1260, 1260)
    
    print(f"  speed_m1: {speed_m1.shape}, speed_m2: {speed_m2.shape}")
    print(f"  test_X_hist: {test_X_hist.shape}")
    print(f"  adjacency: {adj.shape}, edges: {np.count_nonzero(adj)}")
    print(f"  Loaded in {time.time()-t0:.1f}s")
    
    return speed_m1, speed_m2, text_m1, text_m2, test_X_hist, test_texts, adj


# ============================================================
# 2. CREATE SLIDING WINDOWS FROM CONTINUOUS SPEED DATA
# ============================================================
def create_sliding_windows(speed_data, max_horizon=15):
    """
    From continuous speed data (T, N_ROADS), create:
      X: (num_samples, HIST_LEN, N_ROADS) - history windows
      Y: (num_samples, 3, N_ROADS) - targets at h5, h10, h15
    """
    T = speed_data.shape[0]
    num_samples = T - HIST_LEN - max_horizon
    
    X = np.zeros((num_samples, HIST_LEN, N_ROADS), dtype=np.float32)
    Y = np.zeros((num_samples, 3, N_ROADS), dtype=np.float32)
    
    for i in range(num_samples):
        X[i] = speed_data[i : i + HIST_LEN]
        for j, h in enumerate(HORIZONS):
            Y[i, j] = speed_data[i + HIST_LEN - 1 + h]
    
    return X, Y


def get_text_for_windows(text_dict, prefix, num_total, max_horizon=15):
    """Get the text corresponding to the last timestep of each window."""
    num_samples = num_total - HIST_LEN - max_horizon
    texts = []
    for i in range(num_samples):
        # The last timestep index of the window is i + HIST_LEN - 1
        # Text keys are 1-indexed: prefix_1, prefix_2, ...
        key = f"{prefix}_{i + HIST_LEN}"
        texts.append(text_dict.get(key, ""))
    return texts


# ============================================================
# 3. TEXT FEATURE EXTRACTION (TF-IDF + SVD)
# ============================================================
def build_text_features(train_texts_list, test_texts_list, n_components=N_TEXT_FEATURES):
    """
    Fit TF-IDF + SVD on all train texts, transform both train and test.
    Returns: train_text_feats (n_train, n_components), test_text_feats (n_test, n_components), fitted objects
    """
    print("  Building text features (TF-IDF + SVD)...")
    
    all_texts = train_texts_list + test_texts_list
    
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix = tfidf.fit_transform(all_texts)
    
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    
    train_text_feats = svd_matrix[:len(train_texts_list)].astype(np.float32)
    test_text_feats = svd_matrix[len(train_texts_list):].astype(np.float32)
    
    print(f"    TF-IDF vocab: {len(tfidf.vocabulary_)}, SVD explained var: {svd.explained_variance_ratio_.sum():.3f}")
    
    return train_text_feats, test_text_feats


# ============================================================
# 4. FEATURE ENGINEERING (per-road features from window)
# ============================================================
def compute_neighbor_degree(adj):
    """Precompute neighbor indices and degrees from adjacency matrix."""
    neighbor_indices = []
    for r in range(N_ROADS):
        neighbors = np.where(adj[r] > 0)[0]
        neighbor_indices.append(neighbors)
    return neighbor_indices


def extract_features_for_samples(X_windows, text_feats, neighbor_indices, road_ids=None):
    """
    Extract features for a batch of samples, for all roads or specific roads.
    X_windows: (n_samples, 15, 1260) - speed history windows
    text_feats: (n_samples, n_text_components) - text SVD features  
    
    Returns: feature matrix (n_samples * n_roads, n_features)
    """
    n_samples = X_windows.shape[0]
    if road_ids is None:
        road_ids = np.arange(N_ROADS)
    n_roads = len(road_ids)
    
    # Per-road temporal features from history
    # For each road: last speed, mean, std, min, max, diff, trend
    n_temporal = 7 + HIST_LEN   # 7 stats + 15 raw values
    n_spatial = 4               # neighbor mean, std, min, max at last timestep
    n_text = text_feats.shape[1]
    n_static = 2                # road_id, degree
    n_features = n_temporal + n_spatial + n_text + n_static
    
    features = np.zeros((n_samples, n_roads, n_features), dtype=np.float32)
    
    for idx, r in enumerate(road_ids):
        # shape: (n_samples, 15)
        road_hist = X_windows[:, :, r]
        
        col = 0
        # Raw history values (15 features)
        features[:, idx, col:col+HIST_LEN] = road_hist
        col += HIST_LEN
        
        # Stats over the 15 timesteps
        features[:, idx, col] = road_hist[:, -1]                    # last speed
        features[:, idx, col+1] = road_hist.mean(axis=1)            # mean
        features[:, idx, col+2] = road_hist.std(axis=1)             # std
        features[:, idx, col+3] = road_hist.min(axis=1)             # min
        features[:, idx, col+4] = road_hist.max(axis=1)             # max
        features[:, idx, col+5] = road_hist[:, -1] - road_hist[:, 0]  # diff (trend)
        features[:, idx, col+6] = road_hist[:, -1] - road_hist.mean(axis=1)  # deviation from mean
        col += 7
        
        # Spatial: neighbor features at last timestep
        neighbors = neighbor_indices[r]
        if len(neighbors) > 0:
            neigh_speeds = X_windows[:, -1, neighbors]  # (n_samples, n_neighbors)
            features[:, idx, col] = neigh_speeds.mean(axis=1)
            features[:, idx, col+1] = neigh_speeds.std(axis=1)
            features[:, idx, col+2] = neigh_speeds.min(axis=1)
            features[:, idx, col+3] = neigh_speeds.max(axis=1)
        else:
            # No neighbors: use own last speed
            features[:, idx, col:col+4] = road_hist[:, -1:].repeat(4, axis=1)
        col += 4
        
        # Text features (same for all roads in a sample)
        features[:, idx, col:col+n_text] = text_feats
        col += n_text
        
        # Static: road ID and degree
        features[:, idx, col] = r
        features[:, idx, col+1] = len(neighbors)
    
    # Reshape to (n_samples * n_roads, n_features)
    features = features.reshape(-1, n_features)
    
    return features, n_features


# ============================================================
# 5. TRAINING WITH TIME-SERIES CV
# ============================================================
def train_model(X_feat, Y_targets, n_features):
    """
    Train LightGBM models with walk-forward time-series split.
    X_feat: (n_samples * n_roads, n_features) 
    Y_targets: (n_samples, 3, n_roads) - targets for 3 horizons
    
    Returns: list of models (one per horizon per fold)
    """
    print("=" * 60)
    print("[4] Training LightGBM models...")
    
    n_samples = Y_targets.shape[0]
    
    # Walk-forward split with gap
    gap = 15  # gap timesteps to prevent leakage
    fold_size = n_samples // (N_FOLDS + 1)
    
    models = {h: [] for h in range(3)}  # 3 horizons
    oof_scores = {h: [] for h in range(3)}
    
    for fold in range(N_FOLDS):
        val_end = n_samples - (fold * fold_size)
        val_start = val_end - fold_size
        train_end = val_start - gap
        
        if train_end <= 0:
            continue
            
        print(f"\n  Fold {fold+1}/{N_FOLDS}: train [0:{train_end}], val [{val_start}:{val_end}]")
        
        # Training indices (sample-level -> flatten to road-level)
        tr_idx = np.arange(train_end)
        va_idx = np.arange(val_start, val_end)
        
        # Flatten: each sample has N_ROADS rows
        tr_row_idx = np.concatenate([np.arange(s * N_ROADS, (s+1) * N_ROADS) for s in tr_idx])
        va_row_idx = np.concatenate([np.arange(s * N_ROADS, (s+1) * N_ROADS) for s in va_idx])
        
        X_tr = X_feat[tr_row_idx]
        X_va = X_feat[va_row_idx]
        
        for h in range(3):
            # Targets: Y_targets[sample, horizon, road] -> flatten
            y_tr = Y_targets[tr_idx, h, :].reshape(-1)
            y_va = Y_targets[va_idx, h, :].reshape(-1)
            
            model = lgb.LGBMRegressor(**LGB_PARAMS)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(200),
                ],
            )
            
            pred_va = model.predict(X_va)
            mse = np.mean((pred_va - y_va) ** 2)
            oof_scores[h].append(mse)
            models[h].append(model)
            
            print(f"    Horizon h{HORIZONS[h]}: MSE={mse:.4f}, best_iter={model.best_iteration_}")
    
    print("\n  === CV Summary ===")
    for h in range(3):
        scores = oof_scores[h]
        print(f"  h{HORIZONS[h]}: mean MSE = {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    overall = np.mean([np.mean(v) for v in oof_scores.values()])
    print(f"  Overall mean MSE: {overall:.4f}")
    
    return models


# ============================================================
# 6. INFERENCE & SUBMISSION
# ============================================================
def predict_and_submit(models, test_X_hist, test_text_feats, neighbor_indices):
    """
    Run inference on test data and generate submission.csv
    """
    print("=" * 60)
    print("[5] Running inference...")
    t0 = time.time()
    
    n_test = test_X_hist.shape[0]  # 540
    
    # Extract features for test
    test_features, _ = extract_features_for_samples(
        test_X_hist, test_text_feats, neighbor_indices
    )
    print(f"  Test features shape: {test_features.shape}")
    
    # Predict with fold ensembling
    predictions = np.zeros((n_test, 3, N_ROADS), dtype=np.float64)
    
    for h in range(3):
        fold_preds = []
        for model in models[h]:
            pred = model.predict(test_features)  # (n_test * n_roads,)
            fold_preds.append(pred.reshape(n_test, N_ROADS))
        
        # Average across folds
        predictions[:, h, :] = np.mean(fold_preds, axis=0)
    
    # Clip to reasonable speed range
    predictions = np.clip(predictions, 0, 160)
    
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Speed range: [{predictions.min():.1f}, {predictions.max():.1f}]")
    print(f"  Inference done in {time.time()-t0:.1f}s")
    
    # Generate submission
    print("\n  Generating submission.csv...")
    rows = []
    horizon_names = ["h5", "h10", "h15"]
    
    for sample_idx in range(n_test):
        for h_idx, h_name in enumerate(horizon_names):
            for road_idx in range(N_ROADS):
                sample_id = f"test_{sample_idx:05d}_{h_name}_r{road_idx}"
                speed = predictions[sample_idx, h_idx, road_idx]
                rows.append((sample_id, speed))
    
    df = pd.DataFrame(rows, columns=["id", "speed"])
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    df.to_csv(submission_path, index=False)
    
    print(f"  Submission saved: {submission_path}")
    print(f"  Total rows: {len(df)} (expected: {540*3*1260})")
    assert len(df) == 540 * 3 * 1260, f"Row count mismatch! Got {len(df)}"
    
    return df


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    total_start = time.time()
    
    # Step 1: Load
    speed_m1, speed_m2, text_m1, text_m2, test_X_hist, test_texts, adj = load_data()
    
    # Step 2: Create sliding windows
    print("=" * 60)
    print("[2] Creating sliding windows...")
    t0 = time.time()
    
    X1, Y1 = create_sliding_windows(speed_m1)
    X2, Y2 = create_sliding_windows(speed_m2)
    
    texts1 = get_text_for_windows(text_m1, "m1", speed_m1.shape[0])
    texts2 = get_text_for_windows(text_m2, "m2", speed_m2.shape[0])
    
    print(f"  Windows from m1: X={X1.shape}, Y={Y1.shape}, texts={len(texts1)}")
    print(f"  Windows from m2: X={X2.shape}, Y={Y2.shape}, texts={len(texts2)}")
    
    # Combine both months
    X_train = np.concatenate([X1, X2], axis=0)
    Y_train = np.concatenate([Y1, Y2], axis=0)
    train_texts = texts1 + texts2
    
    print(f"  Combined: X={X_train.shape}, Y={Y_train.shape}, texts={len(train_texts)}")
    print(f"  Done in {time.time()-t0:.1f}s")
    
    del speed_m1, speed_m2, X1, X2, Y1, Y2
    gc.collect()
    
    # Step 3: Text features
    print("=" * 60)
    print("[3] Feature engineering...")
    t0 = time.time()
    
    # Prepare test texts (dict -> list in order)
    test_texts_list = [test_texts[f"test_{i:05d}"] for i in range(test_X_hist.shape[0])]
    
    train_text_feats, test_text_feats = build_text_features(train_texts, test_texts_list)
    
    # Neighbor structure from adjacency
    neighbor_indices = compute_neighbor_degree(adj)
    
    # Extract features for all training samples
    print("  Extracting training features...")
    train_features, n_features = extract_features_for_samples(
        X_train, train_text_feats, neighbor_indices
    )
    print(f"  Train features shape: {train_features.shape}, n_features: {n_features}")
    print(f"  Feature engineering done in {time.time()-t0:.1f}s")
    
    del X_train
    gc.collect()
    
    # Step 4: Train
    models = train_model(train_features, Y_train, n_features)
    
    del train_features, Y_train
    gc.collect()
    
    # Step 5: Inference & Submit
    submission = predict_and_submit(models, test_X_hist, test_text_feats, neighbor_indices)
    
    total_time = time.time() - total_start
    print("=" * 60)
    print(f"Pipeline complete! Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Submission file: {os.path.join(OUTPUT_DIR, 'submission.csv')}")


if __name__ == "__main__":
    main()
