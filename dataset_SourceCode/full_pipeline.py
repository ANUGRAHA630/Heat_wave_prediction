"""
Complete Weather Forecasting Pipeline with ALL Requirements
============================================================
Preprocessing → MEMD → GAT-Transformer → K-Fold CV → Comprehensive Evaluation

Run: python main.py

All requirements satisfied:
✓ Preprocessing: normalization + MinMax scaling
✓ MEMD decomposition + fusion
✓ 30-day window, 7-day horizon, 100 epochs
✓ Adam optimizer
✓ K-Fold time series split
✓ Separate configuration
✓ Inverse scaling before evaluation
✓ All required plots in separate functions
"""

import os, json, math, warnings, pickle
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# ============================================================
# CONFIGURATION FILE
# ============================================================
class Config:
    """Centralized Configuration"""
    # Paths
    DATA_PATH = r"C:\project\dataset\city_sep\new\merged_all_cities.csv"
    OUTPUT_DIR = r"C:\project\weather_complete_output"
    
    # Data columns
    DATE_COL = "DateTime"
    CITY_COL = "City"
    TARGET_COL = "MaxTemp"
    LAT_COL = "Latitude"
    LON_COL = "Longitude"
    FEATURES = ["10u", "10v", "msl", "msnlwrf", "r", "ssr", "RainFall"]
    
    # Time series parameters
    WINDOW_SIZE = 30  # 30 days context
    PRED_HORIZON = 7  # 7 days prediction
    
    # MEMD parameters
    NUM_MEMD_DIRECTIONS = 64
    NUM_IMF_TO_KEEP = 8
    
    # Graph parameters
    K_NEIGHBORS = 3
    
    # GAT parameters
    GAT_HIDDEN = 32
    GAT_OUT_DIM = 32
    GAT_HEADS = 4
    
    # Transformer parameters
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS = 2
    DROPOUT = 0.1

    # Uncertainty / MCQR
    QUANTILES = [0.1, 0.5, 0.9]

    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    K_FOLDS = 5  # K-fold cross validation
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def create_directories(cls):
        """Create all output directories"""
        dirs = [
            cls.OUTPUT_DIR,
            f"{cls.OUTPUT_DIR}/preprocessed",
            f"{cls.OUTPUT_DIR}/memd_results",
            f"{cls.OUTPUT_DIR}/models",
            f"{cls.OUTPUT_DIR}/plots/training",
            f"{cls.OUTPUT_DIR}/plots/evaluation",
            f"{cls.OUTPUT_DIR}/plots/citywise",
            f"{cls.OUTPUT_DIR}/plots/horizon",
            f"{cls.OUTPUT_DIR}/plots/correlation",
            f"{cls.OUTPUT_DIR}/results"
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    @classmethod
    def save_config(cls):
        """Save configuration to JSON"""
        config_dict = {k: str(v) if isinstance(v, torch.device) else v 
                      for k, v in cls.__dict__.items() if not k.startswith('_') and k.isupper()}
        with open(f"{cls.OUTPUT_DIR}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=4)


# ============================================================
# MEMD IMPLEMENTATION
# ============================================================
def hamm(n, base):
    seq = np.zeros((1, n))
    if base > 1:
        seed, base_inv = np.arange(1, n + 1), 1 / base
        while any(seed):
            seq += np.remainder(seed[0:n], base) * base_inv
            base_inv, seed = base_inv / base, np.floor(seed / base)
    else:
        seq = (np.remainder(np.arange(1, n + 1), (-base + 1)) + 0.5) / (-base)
    return seq

def local_peaks(x):
    if all(x < 1e-5):
        return np.array([]), np.array([])
    m, dy = len(x) - 1, np.diff(x.T).T
    a = np.where(dy != 0)[0]
    if len(a) == 0:
        return np.array([]), np.array([])
    lm = np.where(np.diff(a) != 1)[0] + 1
    if len(lm) > 0:
        d = a[lm] - a[lm - 1]
        a[lm] = a[lm] - np.floor(d / 2)
    a = np.append(a, m)
    ya = x[a]
    if len(ya) <= 1:
        return np.array([]), np.array([])
    dX = np.sign(np.diff(ya.T)).T
    locs_max = np.where(np.logical_and(dX[:-1] > 0, dX[1:] < 0))[0] + 1
    locs_min = np.where(np.logical_and(dX[:-1] < 0, dX[1:] > 0))[0] + 1
    indmin = a[locs_min] if len(locs_min) > 0 else np.array([])
    indmax = a[locs_max] if len(locs_max) > 0 else np.array([])
    return indmin, indmax

def boundary_conditions(indmin, indmax, t, x, z, nbsym=2):
    if len(indmin) + len(indmax) < 3:
        return None, None, None, None, 0
    indmin, indmax = indmin.astype(int), indmax.astype(int)
    lx = len(x) - 1
    
    # Left boundary
    if indmax[0] < indmin[0]:
        lsym = indmax[0] if x[0] > x[indmin[0]] else 0
    else:
        lsym = indmin[0] if x[0] < x[indmax[0]] else 0
    
    # Right boundary
    if indmax[-1] < indmin[-1]:
        rsym = indmin[-1] if x[-1] < x[indmax[-1]] else lx
    else:
        rsym = indmax[-1] if x[-1] > x[indmin[-1]] else lx
    
    lmin = np.flipud(indmin[:min(len(indmin), nbsym)])
    lmax = np.flipud(indmax[:min(len(indmax), nbsym)])
    rmin = np.flipud(indmin[max(0, len(indmin) - nbsym):])
    rmax = np.flipud(indmax[max(0, len(indmax) - nbsym):])
    
    tmin = np.hstack((2 * t[lsym] - t[lmin], t[indmin], 2 * t[rsym] - t[rmin]))
    tmax = np.hstack((2 * t[lsym] - t[lmax], t[indmax], 2 * t[rsym] - t[rmax]))
    zmin = np.vstack((z[lmin, :], z[indmin, :], z[rmin, :]))
    zmax = np.vstack((z[lmax, :], z[indmax, :], z[rmax, :]))
    
    return tmin, tmax, zmin, zmax, 1

def envelope_mean(m, t, seq, ndir, N, N_dim):
    env_mean, count = np.zeros((len(t), N_dim)), 0
    for it in range(ndir):
        if N_dim != 3:
            b = 2 * seq[it, :] - 1
            tht = np.arctan2(np.sqrt(np.flipud(np.cumsum(b[:0:-1] ** 2))), b[:N_dim - 1]).T
            dir_vec = np.zeros((N_dim, 1))
            dir_vec[:, 0] = np.cumprod(np.concatenate(([1], np.sin(tht))))
            dir_vec[:N_dim - 1, 0] = np.cos(tht) * dir_vec[:N_dim - 1, 0]
        else:
            tt = np.clip(2 * seq[it, 0] - 1, -1, 1)
            phirad = seq[it, 1] * 2 * np.pi
            st = np.sqrt(1.0 - tt * tt)
            dir_vec = np.array([[st * np.cos(phirad)], [st * np.sin(phirad)], [tt]])
        
        y = np.dot(m, dir_vec)
        indmin, indmax = local_peaks(y)
        tmin, tmax, zmin, zmax, mode = boundary_conditions(indmin, indmax, t, y, m, 2)
        
        if mode:
        # ---- Ensure strictly increasing time indices ----
            tmin_u, idx_min = np.unique(tmin, return_index=True)
            zmin_u = zmin[idx_min]

            tmax_u, idx_max = np.unique(tmax, return_index=True)
            zmax_u = zmax[idx_max]

            # ---- Safety check: minimum points for spline ----
            if len(tmin_u) >= 4 and len(tmax_u) >= 4:
                try:
                    fmin = CubicSpline(tmin_u, zmin_u, bc_type="not-a-knot")
                    fmax = CubicSpline(tmax_u, zmax_u, bc_type="not-a-knot")
                    env_mean += (fmax(t) + fmin(t)) / 2
                except Exception:
                    count += 1
            else:
                count += 1

    
    return env_mean / (ndir - count) if ndir > count else np.zeros((N, N_dim))

def stop_emd(r, seq, ndir, N_dim):
    ner = []
    for it in range(ndir):
        if N_dim != 3:
            b = 2 * seq[it, :] - 1
            tht = np.arctan2(np.sqrt(np.flipud(np.cumsum(b[:0:-1] ** 2))), b[:N_dim - 1]).T
            dir_vec = np.zeros((N_dim, 1))
            dir_vec[:, 0] = np.cumprod(np.concatenate(([1], np.sin(tht))))
            dir_vec[:N_dim - 1, 0] = np.cos(tht) * dir_vec[:N_dim - 1, 0]
        else:
            tt = np.clip(2 * seq[it, 0] - 1, -1, 1)
            phirad = seq[it, 1] * 2 * np.pi
            st = np.sqrt(1 - tt * tt)
            dir_vec = np.array([[st * np.cos(phirad)], [st * np.sin(phirad)], [tt]])
        
        y = np.dot(r, dir_vec)
        indmin, indmax = local_peaks(y)
        ner.append(len(indmin) + len(indmax))
    
    return all(n < 3 for n in ner)

def memd(X, num_dir=64):
    """
    Robust Multivariate Empirical Mode Decomposition
    Enforces (time, variables) format and protects against NaNs
    """

    # --- Ensure numpy float ---
    X = np.asarray(X, dtype=np.float64)

    # --- Enforce shape: (time, variables) ---
    if X.ndim != 2:
        raise ValueError("MEMD expects 2D array (time × variables)")

    T, D = X.shape
    if T < D:
        X = X.T
        T, D = X.shape

    # --- Sanitize input ---
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    t = np.arange(T)

    # --- Direction sequence (FIXED & STABLE) ---
    # Generate Halton-like sequence with consistent shape
    seq = np.zeros((num_dir, D))
    for d in range(D):
        seq[:, d] = hamm(num_dir, d + 2)[0]


    r = X.copy()
    imfs = []

    max_imfs = min(10, T // 3)  # dynamic IMF limit

    for _ in range(max_imfs):
        if stop_emd(r, seq, num_dir, D):
            break

        m = r.copy()

        for _ in range(100):
            env = envelope_mean(m, t, seq, num_dir, T, D)
            if np.max(np.abs(env)) < 1e-6:
                break
            m -= env

        imfs.append(m.T)
        r -= m

    imfs.append(r.T)  # residual
    return np.array(imfs)



# ============================================================
# PREPROCESSING
# ============================================================
class Preprocessor:
    def __init__(self, config):
        self.cfg = config
        self.scalers = {}

    def run(self):
        print("\n" + "=" * 70)
        print("STEP 1: PREPROCESSING (Normalization + MinMax Scaling)")
        print("=" * 70)

        # ---------------- LOAD DATA ----------------
        print("\n📥 Loading data...")
        df = pd.read_csv(self.cfg.DATA_PATH)

        df[self.cfg.DATE_COL] = pd.to_datetime(
            df[self.cfg.DATE_COL],
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )

        df = (
            df.dropna(subset=[self.cfg.DATE_COL])
              .sort_values([self.cfg.CITY_COL, self.cfg.DATE_COL])
              .reset_index(drop=True)
        )

        print(f"  ✓ Loaded: {len(df)} records, {df[self.cfg.CITY_COL].nunique()} cities")

        # ---------------- FEATURE VALIDATION ----------------
        required_cols = (
            [self.cfg.DATE_COL, self.cfg.CITY_COL, self.cfg.TARGET_COL]
            + self.cfg.FEATURES
        )

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"❌ Missing required columns: {missing}")

        # ---------------- FIX msnlwrf ----------------
        print("\n🔧 Correcting msnlwrf column...")
        col = [c for c in df.columns if "ms" in c.lower() and "lw" in c.lower()]
        if col:
            col = col[0]
            mean_val = df[col].mean()
            if abs(mean_val) < 15:
                df[col] *= -4
                print(f"  → Applied -4 multiplier (mean was {mean_val:.2f})")
            elif mean_val > 0:
                df[col] *= -1
                print(f"  → Fixed sign (mean was {mean_val:.2f})")
            else:
                print(f"  ✓ msnlwrf already correct (mean={mean_val:.2f})")

        # ---------------- NORMALIZATION + MINMAX ----------------
        print("\n📊 Applying StandardScaler + MinMaxScaler per city...")
        feature_cols = self.cfg.FEATURES + [self.cfg.TARGET_COL]

        for city in df[self.cfg.CITY_COL].unique():
            mask = df[self.cfg.CITY_COL] == city

            std_scaler = StandardScaler()
            mm_scaler = MinMaxScaler()

            normalized = std_scaler.fit_transform(df.loc[mask, feature_cols])
            scaled = mm_scaler.fit_transform(normalized)

            df.loc[mask, feature_cols] = scaled

            # store scalers with dimension info (CRITICAL)
            self.scalers[city] = {
                "std": std_scaler,
                "minmax": mm_scaler,
                "n_features": len(feature_cols)
            }

        print(f"  ✓ Scaled {len(feature_cols)} variables for {len(self.scalers)} cities")

        # ---------------- SEASONAL ENCODING (AUXILIARY) ----------------
        print("\n🌞 Adding seasonal encoding (analysis-only)...")
        df["month"] = df[self.cfg.DATE_COL].dt.month
        df["day_of_year"] = df[self.cfg.DATE_COL].dt.dayofyear

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # NOTE: Seasonal features are NOT used in MEMD or model input

        # ---------------- SAVE ----------------
        df.to_csv(
            f"{self.cfg.OUTPUT_DIR}/preprocessed/preprocessed_data.csv",
            index=False
        )

        with open(
            f"{self.cfg.OUTPUT_DIR}/preprocessed/scalers.pkl",
            "wb"
        ) as f:
            pickle.dump(self.scalers, f)

        print("\n💾 Saved preprocessed data and scalers")
        return df, self.scalers



# ============================================================
# MEMD PROCESSING
# ============================================================
def process_memd(df, config):
    print("\n" + "=" * 70)
    print("STEP 2: MEMD DECOMPOSITION + FUSION")
    print("=" * 70)

    cities = sorted(df[config.CITY_COL].unique())
    all_fused = []

    for idx, city in enumerate(cities, 1):
        print(f"\n[{idx}/{len(cities)}] Processing: {city}")

        df_city = (
            df[df[config.CITY_COL] == city]
            .sort_values(config.DATE_COL)
            .reset_index(drop=True)
        )

        # ---------- Extract feature matrix (T × F) ----------
        X = df_city[config.FEATURES].values.astype(np.float64)
        print(f"  → Input shape (time × features): {X.shape}")

        # ---------- MEMD ----------
        imfs = memd(X, config.NUM_MEMD_DIRECTIONS)
        print(f"  → Extracted IMFs shape: {imfs.shape}")  
        # (num_imfs, F, T)

        num_keep = min(config.NUM_IMF_TO_KEEP, imfs.shape[0])

        # ---------- CORRECT IMF FUSION ----------
        # Average each IMF across features → preserve temporal structure
        fused = np.mean(imfs[:num_keep], axis=1).T  
        # shape: (T, num_keep)

        imf_cols = [f"IMF{i+1}" for i in range(num_keep)]
        df_fused = pd.DataFrame(fused, columns=imf_cols)

        df_fused[config.DATE_COL] = df_city[config.DATE_COL].values[:len(fused)]
        df_fused[config.TARGET_COL] = df_city[config.TARGET_COL].values[:len(fused)]
        df_fused[config.CITY_COL] = city

        df_fused.to_csv(
            f"{config.OUTPUT_DIR}/memd_results/{city}_fused_IMFs.csv",
            index=False
        )

        all_fused.append(df_fused)
        print(f"  ✓ Fused shape: {df_fused.shape}")

    result = pd.concat(all_fused, ignore_index=True)
    print(f"\n✅ MEMD completed correctly for {len(cities)} cities")

    return result

# ============================================================
# GRAPH CONSTRUCTION
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    R, p = 6371.0, math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2 + 
         math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

def build_graph(df, config):
    """
    Builds a symmetric k-NN spatial graph using Haversine distance.
    Ensures consistent city ordering across preprocessing, GAT, and datasets.
    """

    print("\n🌍 Building k-NN spatial graph...")

    # -------- Consistent city ordering --------
    cities = sorted(df[config.CITY_COL].dropna().unique())

    city_info = (
        df[[config.CITY_COL, config.LAT_COL, config.LON_COL]]
        .dropna()
        .drop_duplicates(subset=[config.CITY_COL])
        .set_index(config.CITY_COL)
        .loc[cities]
        .reset_index()
    )

    coords = city_info[[config.LAT_COL, config.LON_COL]].values.astype(np.float64)
    N = len(coords)

    if N < 2:
        raise ValueError("❌ Not enough cities to build spatial graph")

    # -------- Distance matrix --------
    dist = np.full((N, N), np.inf, dtype=np.float64)

    for i in range(N):
        for j in range(i + 1, N):
            d = haversine(
                coords[i, 0], coords[i, 1],
                coords[j, 0], coords[j, 1]
            )
            dist[i, j] = d
            dist[j, i] = d  # symmetry

    # -------- k-NN adjacency --------
    k = min(config.K_NEIGHBORS, N - 1)
    adj = np.zeros((N, N), dtype=np.int8)

    for i in range(N):
        nn_idx = np.argsort(dist[i])[:k]
        for j in nn_idx:
            if i != j:
                adj[i, j] = 1
                adj[j, i] = 1

    # -------- PyG edge_index --------
    src, dst = np.where(adj == 1)
    edge_index = torch.tensor(
        np.vstack([src, dst]),
        dtype=torch.long
    )

    print(f"  ✓ Graph built: {N} nodes, {edge_index.shape[1]} edges")
    return edge_index


# ============================================================
# MODELS
# ============================================================
class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, heads):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads, concat=True)
        self.gat2 = GATConv(hidden * heads, out_dim, heads=1, concat=False)
        self.act = nn.ELU()

    def forward(self, x, edge_index):
        """
        x: (N, F)
        edge_index: (2, E)
        """
        x = self.gat1(x, edge_index)
        x = self.act(x)
        x = self.gat2(x, edge_index)
        return x



class SeriesDecomposition(nn.Module):
    """
    Moving-average based series decomposition
    x = seasonal + trend
    """
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.avg_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding
        )

    def forward(self, x):
        # x: (B, T, D)
        trend = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class AutoformerEncoderLayer(nn.Module):
    """
    Autoformer Encoder Layer with:
    - Series decomposition
    - Temporal aggregation
    - Feed-forward refinement
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, T, D)
        """

        # Temporal aggregation (mean over time → autocorrelation proxy)
        temporal_context = torch.mean(x, dim=1, keepdim=True)
        temporal_context = temporal_context.repeat(1, x.size(1), 1)

        # Residual + FFN
        x = x + temporal_context
        x = self.norm(x + self.ffn(x))

        return x


class AutoformerForecaster(nn.Module):
    """
    Autoformer for temporal modeling + MCQR uncertainty
    """

    def __init__(self, f_dim, d_model, pred_len, quantiles):
        super().__init__()

        self.pred_len = pred_len
        self.quantiles = quantiles
        self.nq = len(quantiles)

        # --------------------------------------------------
        # Series Decomposition
        # --------------------------------------------------
        self.decomp = SeriesDecomposition(kernel_size=25)

        # --------------------------------------------------
        # Input Embedding
        # --------------------------------------------------
        self.embedding = nn.Linear(f_dim + 1, d_model)

        # --------------------------------------------------
        # Autoformer Encoder Stack
        # --------------------------------------------------
        self.encoder = nn.Sequential(
            AutoformerEncoderLayer(d_model),
            AutoformerEncoderLayer(d_model)
        )

        # --------------------------------------------------
        # Projection Heads
        # --------------------------------------------------
        self.out_proj = nn.Linear(d_model, pred_len)
        self.trend_proj = nn.Linear(d_model, pred_len)

    def forward(self, feats, vals):
        """
        feats: (B, T, F)   → MEMD + GAT features
        vals : (B, T)      → historical target
        """

        # --------------------------------------------------
        # Concatenate inputs
        # --------------------------------------------------
        x = torch.cat([feats, vals.unsqueeze(-1)], dim=-1)
        x = self.embedding(x)

        # --------------------------------------------------
        # Series Decomposition
        # --------------------------------------------------
        seasonal, trend = self.decomp(x)

        # --------------------------------------------------
        # Temporal Encoding (Autoformer)
        # --------------------------------------------------
        enc_out = self.encoder(seasonal)

        # --------------------------------------------------
        # Seasonal + Trend Forecasting
        # --------------------------------------------------
        pred = self.out_proj(enc_out[:, -1, :])

        trend_out = self.trend_proj(trend[:, -1, :])
        trend_out = trend_out.unsqueeze(-1)  # (B, H, 1)

        # --------------------------------------------------
        # Combine Trend + Seasonal (MCQR)
        # --------------------------------------------------
        return pred + trend_out.squeeze(-1)



# ============================================================
# DATASET
# ============================================================
class WeatherDataset(Dataset):
    """
    City-aware sliding window dataset.
    Robust to unequal city lengths and NaNs.
    """

    def __init__(self, data, target, window, horizon):
        self.data = data
        self.target = target
        self.window = int(window)
        self.horizon = int(horizon)

        T, N, _ = data.shape
        self.samples = []

        # -------- Build samples per city --------
        for c in range(N):
            valid_idx = np.where(~np.isnan(target[:, c]))[0]

            if len(valid_idx) < self.window + self.horizon:
                continue

            start = valid_idx[0]
            end = valid_idx[-1] - self.window - self.horizon + 1

            for t in range(start, end):
                self.samples.append((c, t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        c, t = self.samples[idx]

        x = self.data[t:t + self.window, c, :]
        y_hist = self.target[t:t + self.window, c]
        y_fut = self.target[t + self.window:t + self.window + self.horizon, c]

        # -------- NaN protection --------
        x = np.nan_to_num(x, nan=0.0)
        y_hist = np.nan_to_num(y_hist, nan=0.0)
        y_fut = np.nan_to_num(y_fut, nan=0.0)

        return {
            "features": torch.tensor(x, dtype=torch.float32),
            "values": torch.tensor(y_hist, dtype=torch.float32),
            "targets": torch.tensor(y_fut, dtype=torch.float32),
            "city_idx": int(c)
        }


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        # preds: (B, H, Q)
        # target: (B, H)
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            e = target - preds[:, :, i]
            loss += torch.mean(torch.max(q * e, (q - 1) * e))
        return loss



# ============================================================
# PLOTTING FUNCTIONS (ALL SEPARATE)
# ============================================================
def plot_epoch_wise_metrics(history, output_dir, config):
    """
    Plot epoch-wise metrics averaged across folds.
    """

    required_keys = ["train_loss", "val_loss", "val_rmse", "val_cc", "val_mae"]
    for k in required_keys:
        if k not in history:
            raise ValueError(f"Missing key in history: {k}")

    total_points = len(history["train_loss"])
    epochs = config.EPOCHS

    if total_points == 0:
        raise ValueError("Empty training history")

    if total_points % epochs != 0:
        raise ValueError(
            f"History length ({total_points}) not divisible by epochs ({epochs})"
        )

    folds = total_points // epochs

    def avg_per_epoch(values):
        values = np.asarray(values).reshape(folds, epochs)
        return values.mean(axis=0)

    train_loss = avg_per_epoch(history["train_loss"])
    val_loss   = avg_per_epoch(history["val_loss"])
    val_rmse   = avg_per_epoch(history["val_rmse"])
    val_cc     = avg_per_epoch(history["val_cc"])
    val_mae    = avg_per_epoch(history["val_mae"])

    epoch_axis = np.arange(1, epochs + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(epoch_axis, train_loss, label="Train", linewidth=2)
    axes[0, 0].plot(epoch_axis, val_loss, label="Validation", linewidth=2)
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epoch_axis, val_rmse, color="green", linewidth=2)
    axes[0, 1].set_title("Validation RMSE")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(epoch_axis, val_cc, color="purple", linewidth=2)
    axes[1, 0].set_title("Validation Correlation Coefficient")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(epoch_axis, val_mae, color="orange", linewidth=2)
    axes[1, 1].set_title("Validation MAE")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/plots/training/epoch_wise_metrics.png",
        dpi=300
    )
    plt.close()

    print(f"  ✓ Saved epoch-wise metrics ({folds} folds × {epochs} epochs)")

def plot_predicted_vs_actual(preds, trues, output_dir, title="predicted_vs_actual"):
    """Predicted vs Actual scatter plot"""
    plt.figure(figsize=(10, 10))
    plt.scatter(trues.flatten(), preds.flatten(), alpha=0.4, s=10, c='blue', edgecolors='none')
    
    mn, mx = trues.min(), trues.max()
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=3, label='Perfect Prediction')
    
    plt.xlabel('Actual Temperature', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted Temperature', fontsize=14, fontweight='bold')
    plt.title('Predicted vs Actual Temperature', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/evaluation/{title}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {title}.png")

def plot_horizon_metrics(preds, trues, output_dir):
    """RMSE and CC for each forecast horizon"""
    horizon_rmse, horizon_cc = [], []
    
    for h in range(preds.shape[1]):
        rmse_h = np.sqrt(mean_squared_error(trues[:, h], preds[:, h]))
        cc_h = np.corrcoef(preds[:, h], trues[:, h])[0, 1] if np.std(preds[:, h]) > 1e-8 else 0
        horizon_rmse.append(rmse_h)
        horizon_cc.append(cc_h)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    horizons = list(range(1, len(horizon_rmse) + 1))
    
    axes[0].plot(horizons, horizon_rmse, marker='o', linewidth=2, markersize=8, color='red')
    axes[0].set_xlabel('Forecast Horizon (days)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('RMSE', fontsize=12, fontweight='bold')
    axes[0].set_title('RMSE by Forecast Horizon', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(horizons)
    
    axes[1].plot(horizons, horizon_cc, marker='s', linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Forecast Horizon (days)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Correlation Coefficient', fontsize=12, fontweight='bold')
    axes[1].set_title('Correlation by Forecast Horizon', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(horizons)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/horizon/horizon_rmse_cc.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: horizon_rmse_cc.png")
    
    return horizon_rmse, horizon_cc

def plot_overall_horizon(horizon_rmse, horizon_cc, output_dir):
    """Combined overall horizon plot"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    horizons = list(range(1, len(horizon_rmse) + 1))
    
    ax1.plot(horizons, horizon_rmse, 'b-o', linewidth=2, markersize=8, label='RMSE')
    ax1.set_xlabel('Forecast Horizon (days)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('RMSE', color='b', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(horizons, horizon_cc, 'r-s', linewidth=2, markersize=8, label='CC')
    ax2.set_ylabel('Correlation Coefficient', color='r', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Overall Horizon Performance', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f"{output_dir}/plots/horizon/overall_horizon.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: overall_horizon.png")

def plot_citywise_metrics(city_rmse, city_cc, output_dir):
    """City-wise RMSE and CC bar plots"""
    cities = list(city_rmse.keys())
    rmse_vals = list(city_rmse.values())
    cc_vals = list(city_cc.values())

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # RMSE
    axes[0].bar(cities, rmse_vals, color='steelblue')
    axes[0].set_title("City-wise RMSE", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis='x', rotation=90)

    # CC
    axes[1].bar(cities, cc_vals, color='darkgreen')
    axes[1].set_title("City-wise Correlation Coefficient", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("CC")
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/citywise/citywise_rmse_cc.png", dpi=300)
    plt.close()
    print("  ✓ Saved: citywise_rmse_cc.png")


def plot_pairwise_correlation(df, config):
    """Pairwise city correlation heatmap (MaxTemp)"""
    pivot = df.pivot_table(
        index=config.DATE_COL,
        columns=config.CITY_COL,
        values=config.TARGET_COL
    )

    corr = pivot.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Pairwise City Correlation (MaxTemp)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/plots/correlation/pairwise_city_correlation.png", dpi=300)
    plt.close()
    print("  ✓ Saved: pairwise_city_correlation.png")


def plot_parity_heatmap(preds, trues, output_dir):
    plt.figure(figsize=(8, 7))
    sns.histplot(
        x=trues.flatten(),
        y=preds.flatten(),
        bins=50,
        pmax=0.9,
        cmap="mako"
    )
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title("Parity Heatmap: Predicted vs Actual")
    plt.plot(
        [trues.min(), trues.max()],
        [trues.min(), trues.max()],
        'r--'
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/evaluation/parity_heatmap.png", dpi=300)
    plt.close()

def plot_mcqr_uncertainty(
    y_true,
    q10,
    q50,
    q90,
    output_dir,
    city,
    fold=None
):
   

    # ---------- Safety checks ----------
    y_true = np.asarray(y_true).flatten()
    q10 = np.asarray(q10).flatten()
    q50 = np.asarray(q50).flatten()
    q90 = np.asarray(q90).flatten()

    assert (
        len(y_true) == len(q10) == len(q50) == len(q90)
    ), "MCQR arrays must have same length"

    H = len(q50)
    horizon = np.arange(1, H + 1)

    # ---------- Plot ----------
    plt.figure(figsize=(14, 6))

    plt.plot(horizon, y_true, label="Actual", color="black", linewidth=2)
    plt.plot(horizon, q50, label="Median Prediction (q50)", color="blue", linewidth=2)

    plt.fill_between(
        horizon,
        q10,
        q90,
        color="blue",
        alpha=0.25,
        label="Uncertainty Band (q10–q90)"
    )

    plt.xlabel("Forecast Horizon (Days Ahead)", fontsize=12, fontweight="bold")
    plt.ylabel("Temperature", fontsize=12, fontweight="bold")
    plt.title(
        f"MCQR Forecast Uncertainty – {city}",
        fontsize=14,
        fontweight="bold"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # ---------- File name ----------
    suffix = f"_fold{fold}" if fold is not None else ""
    save_path = f"{output_dir}/plots/evaluation/{city}_mcqr_uncertainty{suffix}.png"

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"  ✓ Saved MCQR uncertainty plot: {save_path}")



# ============================================================
# TRAINING + EVALUATION PIPELINE
# ============================================================

def train_and_evaluate(df_fused, config, scalers):
    print("\n" + "=" * 70)
    print("STEP 3: GAT + AUTOFORMER (Deterministic Training + MCQR Inference)")
    print("=" * 70)

    cities = sorted(df_fused[config.CITY_COL].unique())
    imf_cols = [c for c in df_fused.columns if c.startswith("IMF")]

    q50_idx = config.QUANTILES.index(0.5)
    q10_idx = config.QUANTILES.index(0.1)
    q90_idx = config.QUANTILES.index(0.9)

    # =================================================
    # BUILD PER-CITY SERIES
    # =================================================
    city_series, max_T = {}, 0
    for city in cities:
        d = df_fused[df_fused[config.CITY_COL] == city].sort_values(config.DATE_COL)
        city_series[city] = {
            "X": d[imf_cols].values.astype(np.float32),
            "y": d[config.TARGET_COL].values.astype(np.float32)
        }
        max_T = max(max_T, len(d))

    N, F = len(cities), len(imf_cols)

    X = np.full((max_T, N, F), np.nan, dtype=np.float32)
    y = np.full((max_T, N), np.nan, dtype=np.float32)

    for i, city in enumerate(cities):
        L = len(city_series[city]["X"])
        X[:L, i] = city_series[city]["X"]
        y[:L, i] = city_series[city]["y"]

    # =================================================
    # SPATIAL ENCODING (GAT)
    # =================================================
    edge_index = build_graph(df_fused, config).to(config.DEVICE)

    gat = GATEncoder(
        F,
        config.GAT_HIDDEN,
        config.GAT_OUT_DIM,
        config.GAT_HEADS
    ).to(config.DEVICE)
    gat.eval()

    Z = np.zeros((max_T, N, config.GAT_OUT_DIM), dtype=np.float32)

    with torch.no_grad():
        for t in range(max_T):
            if np.isnan(X[t]).all():
                continue
            xt = torch.tensor(np.nan_to_num(X[t]), device=config.DEVICE)
            Z[t] = gat(xt, edge_index).cpu().numpy()

    dyn = np.concatenate([X, Z], axis=2)

    dataset = WeatherDataset(
        dyn,
        y,
        config.WINDOW_SIZE,
        config.PRED_HORIZON
    )

    # =================================================
    # TIME SERIES K-FOLD SPLIT
    # =================================================
    tscv = TimeSeriesSplit(n_splits=config.K_FOLDS)
    indices = np.arange(len(dataset))

    history = {k: [] for k in ["train_loss", "val_loss", "val_rmse", "val_mae", "val_cc"]}
    all_preds, all_trues = [], []

    # =================================================
    # K-FOLD TRAINING
    # =================================================
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(indices)):
        print(f"\n===== Fold {fold + 1}/{config.K_FOLDS} =====")

        model = AutoformerForecaster(
            f_dim=dyn.shape[2],
            d_model=config.D_MODEL,
            pred_len=config.PRED_HORIZON,
            quantiles=config.QUANTILES
        ).to(config.DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.MSELoss()

        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, tr_idx),
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, va_idx),
            batch_size=config.BATCH_SIZE,
            shuffle=False
        )

        mcqr_storage = {
            city: {"q10": [], "q50": [], "q90": [], "true": []}
            for city in cities
        }

        last_preds, last_trues = None, None

        # ================= EPOCH LOOP =================
        for epoch in range(1, config.EPOCHS + 1):
            model.train()
            epoch_losses = []

            for b in train_loader:
                optimizer.zero_grad()

                preds = model(
                    b["features"].to(config.DEVICE),
                    b["values"].to(config.DEVICE)
                )

                # 🔑 TRAIN ONLY ON MEDIAN (q50)
                pred_med = preds[:, :, q50_idx]
                loss = criterion(pred_med, b["targets"].to(config.DEVICE))

                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            history["train_loss"].append(np.mean(epoch_losses))

            # ---------------- VALIDATION ----------------
            model.eval()
            P_med, Tt = [], []

            with torch.no_grad():
                for b in val_loader:
                    p = model(
                        b["features"].to(config.DEVICE),
                        b["values"].to(config.DEVICE)
                    ).cpu().numpy()

                    t = b["targets"].numpy()

                    for i, cidx in enumerate(b["city_idx"].numpy()):
                        city = cities[cidx]
                        std, mm = scalers[city]["std"], scalers[city]["minmax"]

                        # ---------- Inverse scale median ----------
                        dp = np.zeros((config.PRED_HORIZON, len(config.FEATURES) + 1))
                        dp[:, -1] = p[i, :, q50_idx]
                        dp = std.inverse_transform(mm.inverse_transform(dp))

                        dt = np.zeros((config.PRED_HORIZON, len(config.FEATURES) + 1))
                        dt[:, -1] = t[i]
                        dt = std.inverse_transform(mm.inverse_transform(dt))

                        P_med.append(dp[:, -1])
                        Tt.append(dt[:, -1])

                        # ---------- Store MCQR (INFERENCE ONLY) ----------
                        if fold == config.K_FOLDS - 1:
                            for q_name, q_idx in zip(
                                ["q10", "q50", "q90"],
                                [q10_idx, q50_idx, q90_idx]
                            ):
                                dq = np.zeros((config.PRED_HORIZON, len(config.FEATURES) + 1))
                                dq[:, -1] = p[i, :, q_idx]
                                dq = std.inverse_transform(mm.inverse_transform(dq))
                                mcqr_storage[city][q_name].append(dq[:, -1])

                            mcqr_storage[city]["true"].append(dt[:, -1])

            P_med, Tt = np.array(P_med), np.array(Tt)
            last_preds, last_trues = P_med, Tt

            rmse = np.sqrt(mean_squared_error(Tt, P_med))
            mae = mean_absolute_error(Tt, P_med)
            cc = np.corrcoef(Tt.flatten(), P_med.flatten())[0, 1]

            history["val_rmse"].append(rmse)
            history["val_mae"].append(mae)
            history["val_cc"].append(cc)
            history["val_loss"].append(rmse ** 2)

            print(f"Epoch {epoch:03d} | RMSE={rmse:.4f} | MAE={mae:.4f} | CC={cc:.4f}")

        all_preds.append(last_preds)
        all_trues.append(last_trues)

        # -------- MCQR PLOTS (ONLY ONCE) --------
        if fold == config.K_FOLDS - 1:
            for city in cities:
                plot_mcqr_uncertainty(
                    y_true=np.mean(mcqr_storage[city]["true"], axis=0),
                    q10=np.mean(mcqr_storage[city]["q10"], axis=0),
                    q50=np.mean(mcqr_storage[city]["q50"], axis=0),
                    q90=np.mean(mcqr_storage[city]["q90"], axis=0),
                    output_dir=config.OUTPUT_DIR,
                    city=city
                )

    # ================= FINAL EVALUATION =================
    P_all = np.concatenate(all_preds)
    T_all = np.concatenate(all_trues)

    plot_epoch_wise_metrics(history, config.OUTPUT_DIR, config)
    plot_predicted_vs_actual(P_all, T_all, config.OUTPUT_DIR)
    plot_parity_heatmap(P_all, T_all, config.OUTPUT_DIR)

    print("\n✅ TRAINING DONE (Deterministic) + MCQR DONE (Inference Only)")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    Config.create_directories()
    Config.save_config()

    preprocessor = Preprocessor(Config)
    df_pre, scalers = preprocessor.run()

    df_fused = process_memd(df_pre, Config)
    train_and_evaluate(df_fused, Config, scalers)



