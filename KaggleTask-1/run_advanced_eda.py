import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from statsmodels.tsa.stattools import pacf, acf, adfuller
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs("eda_plots", exist_ok=True)

print("Loading data for advanced EDA...")
speed_m1 = np.load("Dataset/train/train_speed_m1_1_11160.npy") # (11160, 1260)
adj = np.load("Dataset/static/matrix.npy")

print("1. Target Distribution & Skewness Analysis")
valid_speeds = speed_m1[speed_m1 > 0] # Exclude zeros for target distribution
plt.figure(figsize=(10, 5))
sns.histplot(valid_speeds.flatten(), bins=100, kde=True, color='purple')
plt.title("Distribusi Kecepatan (Tanpa Anomali 0) - Evaluasi Target MSE", fontsize=14, fontweight='bold')
plt.xlabel("Speed (km/h)")
plt.ylabel("Frekuensi")
plt.savefig("eda_plots/advanced_target_dist.png", dpi=300, bbox_inches="tight")
plt.close()

skewness = pd.Series(valid_speeds.flatten()).skew()
print(f"Skewness of non-zero speeds: {skewness:.3f} (Values > 1 indicate highly skewed)")

print("2. PACF Analysis for Critical Roads (Stationarity & Lag Selection)")
# Pick a high variance road
variances = np.var(speed_m1, axis=0)
critical_road_idx = np.argmax(variances)
ts = speed_m1[:, critical_road_idx]

# Check Stationarity (ADF)
adf_result = adfuller(ts)
print(f"ADF Statistic for Critical Road {critical_road_idx}: {adf_result[0]:.3f}, p-value: {adf_result[1]:.4e}")

# PACF up to 60 lags (60 * 4 mins = 240 mins)
pacf_vals = pacf(ts, nlags=60)
plt.figure(figsize=(12, 4))
plt.bar(range(len(pacf_vals)), pacf_vals, color='teal')
plt.axhline(0, color='black', lw=1)
plt.axhline(0.05, color='red', linestyle='--', alpha=0.5)
plt.axhline(-0.05, color='red', linestyle='--', alpha=0.5)
plt.title(f"PACF for Critical Road {critical_road_idx} (Mencari Lag Optimal)", fontsize=14, fontweight='bold')
plt.xlabel("Lag (x 4 menit)")
plt.ylabel("Partial Autocorrelation")
plt.savefig("eda_plots/advanced_pacf.png", dpi=300, bbox_inches="tight")
plt.close()

print("3. Lagged Spatial Cross-Correlation")
# Find a neighbor of critical_road_idx
neighbors = np.where(adj[critical_road_idx] > 0)[0]
if len(neighbors) > 0:
    neighbor_idx = neighbors[0]
    ts_critical = ts
    ts_neighbor = speed_m1[:, neighbor_idx]
    
    lags = range(-10, 11)
    cross_corr = [pd.Series(ts_critical).corr(pd.Series(ts_neighbor).shift(l)) for l in lags]
    
    plt.figure(figsize=(10, 5))
    plt.plot(lags, cross_corr, marker='o', color='darkorange')
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f"Cross-Correlation: Road {critical_road_idx} vs Neighbor {neighbor_idx}", fontsize=14, fontweight='bold')
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.savefig("eda_plots/advanced_cross_corr.png", dpi=300, bbox_inches="tight")
    plt.close()

# Create advanced_eda.ipynb
nb = new_notebook()
cells = []

cells.append(new_markdown_cell("# Advanced EDA for Traffic Forecasting (MSE Optimization)\nBerdasarkan rekomendasi di `eda_evaluation.md`, notebook ini memuat analisis spesifik untuk arsitektur pemodelan."))

cells.append(new_markdown_cell("## 1. Target Distribution\nMemeriksa skewness untuk memutuskan apakah transformasi log atau robust loss diperlukan."))
cells.append(new_code_cell("""
from IPython.display import Image
Image(filename='eda_plots/advanced_target_dist.png')
"""))

cells.append(new_markdown_cell("## 2. PACF & Stationarity\nMenentukan apakah historical window 15-step sudah cukup. PACF signifikan di lag > 15 berarti kita butuh window lebih panjang."))
cells.append(new_code_cell("""
Image(filename='eda_plots/advanced_pacf.png')
"""))

cells.append(new_markdown_cell("## 3. Lagged Spatial Cross-Correlation\nMelihat seberapa lambat/cepat rambatan kemacetan antar jalan. Jika puncak korelasi bukan di lag 0, berarti spatial feature butuh shifting (Time-Lagged Graph)."))
cells.append(new_code_cell("""
Image(filename='eda_plots/advanced_cross_corr.png')
"""))

nb.cells = cells
with open("advanced_eda.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Advanced EDA completed. Created advanced_eda.ipynb and plots.")
