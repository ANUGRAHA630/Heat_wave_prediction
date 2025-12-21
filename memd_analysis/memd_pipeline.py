import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MEMD_all import memd   # MEMD implementation

# ================== CONFIG ==================
DATA_PATH = r"C:\project\dataset\city_sep\new\merged_all_cities.csv"
OUT_ROOT = r"C:\project\MEMD_analysis"

DATE_COL   = "DateTime"
CITY_COL   = "City"
TARGET_COL = "MaxTemp"

FEATURES = ["10u", "10v", "msl", "msnlwrf", "r", "ssr", "RainFall"]
NUM_IMF_TO_KEEP = 8
DATE_FORMAT = "%d-%m-%Y"
# ============================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# 🔹 Plot fused IMFs – frequency visualization (same Y-axis)
def plot_fused_imfs(city_name: str, df_fused: pd.DataFrame):
    plot_folder = os.path.join(OUT_ROOT, "plots")
    ensure_dir(plot_folder)

    imf_cols = [col for col in df_fused.columns if col.startswith("IMF")]

    ymin = df_fused[imf_cols].min().min()
    ymax = df_fused[imf_cols].max().max()

    plt.figure(figsize=(18, 22))

    for i, col in enumerate(imf_cols, start=1):
        plt.subplot(len(imf_cols), 1, i)
        plt.plot(df_fused[DATE_COL], df_fused[col], linewidth=0.7)
        plt.title(f"{city_name} – {col}", fontsize=10)
        plt.ylim(ymin, ymax)

        if i < len(imf_cols):
            plt.xticks([])

    plt.tight_layout()
    save_path = os.path.join(plot_folder, f"{city_name}_Fused_IMFs.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    📊 Fused IMF plot saved → {save_path}")


# 🔹 Plot individual feature IMFs
def plot_individual_imfs(city_path: str, df_list: list, city_name: str):
    plot_folder = os.path.join(OUT_ROOT, "plots_individual")
    ensure_dir(plot_folder)

    fig, axes = plt.subplots(NUM_IMF_TO_KEEP, 1, figsize=(18, 22), sharex=True)
    
    for i, df_imf in enumerate(df_list):
        cols = [c for c in df_imf.columns if c.startswith("IMF")]
        merged = df_imf[cols].mean(axis=1)
        axes[i].plot(df_imf[DATE_COL], merged, linewidth=0.7)
        axes[i].set_ylabel(f"IMF {i+1}")

    axes[-1].set_xlabel("Time")
    plt.suptitle(f"{city_name} - Average Feature IMFs", fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(plot_folder, f"{city_name}_Feature_IMFs.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    📈 Feature IMF plot saved → {save_path}")


def run_memd_for_city(city_name: str, df_city: pd.DataFrame):
    print(f"\n===== Processing city: {city_name} =====")

    df_city = df_city.sort_values(DATE_COL).dropna(subset=FEATURES + [TARGET_COL]).reset_index(drop=True)
    if df_city.empty:
        print(f"⚠ Skipping {city_name} — insufficient data")
        return

    X = df_city[FEATURES].to_numpy().T
    num_channels, T = X.shape
    print(f"  → Signal: {num_channels} channels × {T} timestamps")

    # Check skip to avoid reprocessing
    city_folder = os.path.join(OUT_ROOT, city_name)
    if os.path.exists(city_folder):
        print(f"  ✔ Already processed → Skipping {city_name}")
        return

    ensure_dir(city_folder)

    print("  → Running MEMD... (wait)")
    imfs = memd(X)
    print(f"  → IMFs extracted: {imfs.shape}")

    num_imf = min(NUM_IMF_TO_KEEP, imfs.shape[0])

    df_imf_list = []

    # Save individual feature IMFs
    for k in range(num_imf):
        imf_k = imfs[k].T
        col_names = [f"IMF{k+1}_{feat}" for feat in FEATURES]
        df_imf = pd.DataFrame(imf_k, columns=col_names)
        df_imf[DATE_COL] = df_city[DATE_COL].iloc[:T].values
        df_imf[TARGET_COL] = df_city[TARGET_COL].iloc[:T].values

        save_path = os.path.join(city_folder, f"IMF_{k+1}.csv")
        df_imf.to_csv(save_path, index=False)
        df_imf_list.append(df_imf)
        print(f"    ✔ Saved IMF {k+1}: {save_path}")

    # Fused IMFs
    fused = imfs[:num_imf].mean(axis=1).T
    fused_cols = [f"IMF{i+1}" for i in range(num_imf)]
    df_fused = pd.DataFrame(fused, columns=fused_cols)
    df_fused[DATE_COL] = df_city[DATE_COL].iloc[:T].values
    df_fused[TARGET_COL] = df_city[TARGET_COL].iloc[:T].values
    df_fused[CITY_COL] = city_name

    fused_folder = os.path.join(OUT_ROOT, "fused")
    ensure_dir(fused_folder)
    fused_path = os.path.join(fused_folder, f"{city_name}_fused_IMFs.csv")
    df_fused.to_csv(fused_path, index=False)
    print(f"    ✔ Saved fused IMF: {fused_path}")

    # Plot both types
    plot_individual_imfs(city_folder, df_imf_list, city_name)
    plot_fused_imfs(city_name, df_fused)


def main():
    ensure_dir(OUT_ROOT)

    print("📥 Loading dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format=DATE_FORMAT, errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values([CITY_COL, DATE_COL])

    cities = sorted(df[CITY_COL].unique())
    print(f"→ Found {len(cities)} cities:", cities)

    for city in cities:
        run_memd_for_city(city, df[df[CITY_COL] == city].copy())

    print("\n🎯 Completed MEMD + Visual Output for all cities!")
    print("📂 Saved in:", OUT_ROOT)


if __name__ == "__main__":
    main()
