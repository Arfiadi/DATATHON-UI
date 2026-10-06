import numpy as np
import json
import os
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import math

start_time = time.time()

base_dir = r"D:\ARFI\Lomba\datathon-ui\Datathon-UI\Dataset"

print("1. Loading Data...")
train_speed = np.load(os.path.join(base_dir, "train", "train_speed_m1_1_11160.npy")) # (11160, 1260)
with open(os.path.join(base_dir, "train", "train_text_m1_1_11160.json"), 'r') as f:
    train_text = json.load(f)

# Subset data to speed up training (e.g. use last 3000 timesteps of month 1 to simulate closer to test distribution)
N_steps = train_speed.shape[0]
START_STEP = max(0, N_steps - 3000) # Use 3000 timesteps ~ 8.3 days
train_speed = train_speed[START_STEP:]
# Train text subset
keys = list(train_text.keys())
train_text = {k: train_text[k] for k in keys[START_STEP:START_STEP + 3000]}

print(f"Using {train_speed.shape[0]} timesteps for fast training.")

# TF-IDF on Text
print("2. Processing NLP Features...")
text_list = []
for k in train_text.values():
    text_list.append(k if k else "")

tfidf = TfidfVectorizer(max_features=5, stop_words='english')
tfidf_feats = tfidf.fit_transform(text_list).toarray() # (3000, 5)

print("3. Building Tabular Dataset for LightGBM...")
# Sliding window config
history_len = 15
h5, h10, h15 = 5, 10, 15
max_lookahead = 15

X_rows = []
Y5_rows, Y10_rows, Y15_rows = [], [], []

# Generate sliding windows
n_roads = 1260
num_windows = train_speed.shape[0] - history_len - max_lookahead + 1

# Subsample roads if memory is an issue? Let's try all roads for now.
# To vectorize, we can construct the arrays directly.
print(f"Creating windows... Total expected windows: {num_windows}")

# vectorized approach
X_list = []
y5_list = []
y10_list = []
y15_list = []
road_ids = []
time_ids = []
nlp_feats = []

for w in range(num_windows):
    # History: w to w+14
    hist = train_speed[w : w + history_len, :] # (15, 1260)
    # Targets
    t5 = train_speed[w + history_len + h5 - 1, :] # (1260,)
    t10 = train_speed[w + history_len + h10 - 1, :]
    t15 = train_speed[w + history_len + h15 - 1, :]
    
    # NLP for this window (using the last timestep of the history window)
    nlp = tfidf_feats[w + history_len - 1, :] # (5,)
    
    # Append to lists
    X_list.append(hist.T) # (1260, 15)
    y5_list.append(t5) # (1260,)
    y10_list.append(t10)
    y15_list.append(t15)
    road_ids.append(np.arange(n_roads))
    time_ids.append(np.full(n_roads, w))
    nlp_feats.append(np.tile(nlp, (n_roads, 1))) # (1260, 5)

X_all = np.concatenate(X_list, axis=0) # (num_windows * 1260, 15)
y5_all = np.concatenate(y5_list, axis=0)
y10_all = np.concatenate(y10_list, axis=0)
y15_all = np.concatenate(y15_list, axis=0)
road_all = np.concatenate(road_ids, axis=0)
time_all = np.concatenate(time_ids, axis=0)
nlp_all = np.concatenate(nlp_feats, axis=0)

# Create temporal features
# Assuming 1 timestep = 4 minutes (1 hour = 15 timesteps, 1 day = 360 timesteps)
time_in_day = time_all % 360
hour_sin = np.sin(2 * np.pi * time_in_day / 360)
hour_cos = np.cos(2 * np.pi * time_in_day / 360)

# Construct final feature matrix
features = np.column_stack([
    road_all,
    hour_sin,
    hour_cos,
    X_all,
    nlp_all
])
# Column names
feature_names = ['road_id', 'hour_sin', 'hour_cos'] + [f'lag_{i}' for i in range(1, 16)] + [f'nlp_{i}' for i in range(5)]

print(f"Feature matrix shape: {features.shape}")

print("4. Training LightGBM Models...")
params = {
    'objective': 'regression_l2',
    'metric': 'mse',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'n_estimators': 50, # Keep it small for speed
    'verbose': -1,
    'n_jobs': -1
}

# H5 Model
print("Training H5 Model...")
model_h5 = lgb.LGBMRegressor(**params)
model_h5.fit(features, y5_all, categorical_feature=['road_id'])

# H10 Model
print("Training H10 Model...")
model_h10 = lgb.LGBMRegressor(**params)
model_h10.fit(features, y10_all, categorical_feature=['road_id'])

# H15 Model
print("Training H15 Model...")
model_h15 = lgb.LGBMRegressor(**params)
model_h15.fit(features, y15_all, categorical_feature=['road_id'])

print("5. Processing Test Data & Inference...")
test_X_hist = np.load(os.path.join(base_dir, "test", "test_X_hist.npy")) # (540, 15, 1260)
with open(os.path.join(base_dir, "test", "test_texts.json"), 'r') as f:
    test_texts = json.load(f)

test_text_list = []
for k in test_texts.values():
    test_text_list.append(k if k else "")
test_nlp = tfidf.transform(test_text_list).toarray() # (540, 5)

num_test = test_X_hist.shape[0]

submission_rows = []

# Infer row by row (or we can vectorize)
test_X_flat = []
test_nlp_flat = []
test_road_flat = []

for i in range(num_test):
    hist = test_X_hist[i] # (15, 1260)
    nlp = test_nlp[i] # (5,)
    
    test_X_flat.append(hist.T) # (1260, 15)
    test_nlp_flat.append(np.tile(nlp, (n_roads, 1))) # (1260, 5)
    test_road_flat.append(np.arange(n_roads))

test_X_all = np.concatenate(test_X_flat, axis=0) # (540 * 1260, 15)
test_nlp_all = np.concatenate(test_nlp_flat, axis=0)
test_road_all = np.concatenate(test_road_flat, axis=0)

# Dummy time for test (we don't know exactly when test starts, default to 0)
# To be safer, we just use 0, or we could estimate from data.
test_hour_sin = np.zeros(test_X_all.shape[0])
test_hour_cos = np.zeros(test_X_all.shape[0])

test_features = np.column_stack([
    test_road_all,
    test_hour_sin,
    test_hour_cos,
    test_X_all,
    test_nlp_all
])

print("Predicting...")
pred_h5 = model_h5.predict(test_features)
pred_h10 = model_h10.predict(test_features)
pred_h15 = model_h15.predict(test_features)

# Clip negative values
pred_h5 = np.clip(pred_h5, 0, 150)
pred_h10 = np.clip(pred_h10, 0, 150)
pred_h15 = np.clip(pred_h15, 0, 150)

print("6. Formatting Submission...")
# Reshape predictions back to (540, 1260)
pred_h5 = pred_h5.reshape(num_test, n_roads)
pred_h10 = pred_h10.reshape(num_test, n_roads)
pred_h15 = pred_h15.reshape(num_test, n_roads)

submission = []
test_keys = list(test_texts.keys())
for i in range(num_test):
    sample_id = test_keys[i]
    for r in range(n_roads):
        # Format: test_{sample}_h{horizon}_r{road}
        submission.append([f"{sample_id}_h5_r{r}", pred_h5[i, r]])
        submission.append([f"{sample_id}_h10_r{r}", pred_h10[i, r]])
        submission.append([f"{sample_id}_h15_r{r}", pred_h15[i, r]])

sub_df = pd.DataFrame(submission, columns=['id', 'speed'])
out_path = os.path.join(base_dir, "submission_lgbm.csv")
sub_df.to_csv(out_path, index=False)
print(f"Submission saved to {out_path}")
print(f"Total pipeline time: {time.time() - start_time:.2f} seconds.")
