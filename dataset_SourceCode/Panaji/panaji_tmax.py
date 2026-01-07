import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# === Folders ===
input_folder = r"C:\Users\tjpri\Downloads\dataset\maxtemp\tmax_data\maxtemp"
output_folder = r"C:\Users\tjpri\Downloads\dataset\maxtemp\outputs_tmax_panaji"
os.makedirs(output_folder, exist_ok=True)

# === Panaji coordinates ===
target_lat, target_lon = 15.49, 73.83   # Panaji, Goa

# === IMD 1° grid setup ===
lat_start, lon_start = 6.5, 66.5
lat_step, lon_step = 1.0, 1.0
ny, nx = 31, 31

lats = np.round(np.arange(lat_start, lat_start + ny * lat_step, lat_step), 2)
lons = np.round(np.arange(lon_start, lon_start + nx * lon_step, lon_step), 2)


# === Find nearest valid pixel ===
def find_nearest_valid_pixel(data, lats, lons, lat, lon):
    lat_idx = np.argmin(np.abs(lats - lat))
    lon_idx = np.argmin(np.abs(lons - lon))

    for radius in range(0, 6):
        lat_min = max(0, lat_idx - radius)
        lat_max = min(len(lats), lat_idx + radius + 1)
        lon_min = max(0, lon_idx - radius)
        lon_max = min(len(lons), lon_idx + radius + 1)

        subgrid = data[:, lat_min:lat_max, lon_min:lon_max]
        valid_mask = (subgrid != 99.9) & (~np.isnan(subgrid))

        if np.any(valid_mask):
            valid_pos = np.argwhere(valid_mask)[0]
            return lat_min + valid_pos[1], lon_min + valid_pos[2]

    raise ValueError("No valid pixel found.")


all_data = []

# === Process year-wise files ===
for file in sorted(os.listdir(input_folder)):
    if not file.endswith(".GRD"):
        continue

    # Extract year from filename
    year = int("".join(ch for ch in file if ch.isdigit())[:4] or 0)
    if year < 1979 or year > 2024:
        continue

    print(f"\nProcessing year {year} ...")
    path = os.path.join(input_folder, file)

    # Read GRD file
    raw = np.fromfile(path, dtype=np.float32)
    ndays = int(raw.size / (ny * nx))
    data = raw.reshape(ndays, ny, nx)

    # Replace missing values
    data[data >= 99.9] = np.nan

    # Locate nearest usable grid cell
    try:
        lat_idx, lon_idx = find_nearest_valid_pixel(
            data, lats, lons, target_lat, target_lon
        )
        print(f"✅ Using nearest valid grid: ({lats[lat_idx]}°N, {lons[lon_idx]}°E)")
    except ValueError:
        print(f"⚠️ No valid pixel for {year}")
        continue

    # Extract Tmax series
    tmax = data[:, lat_idx, lon_idx]

    # Date list
    dates = [datetime(year, 1, 1) + timedelta(days=i) for i in range(ndays)]

    # Store
    df = pd.DataFrame({"Date": dates, "Tmax_C": tmax})
    all_data.append(df)


# === Merge all years ===
if all_data:
    final = pd.concat(all_data)
    final.dropna(inplace=True)
    final["Date"] = pd.to_datetime(final["Date"]).dt.strftime("%Y-%m-%d")

    out_csv = os.path.join(
        output_folder,
        "Panaji_Tmax_Daily_1979_2024.csv"
    )

    final.to_csv(out_csv, index=False, float_format="%.2f")
    print(f"\n🎉 Saved Panaji Tmax (1979–2024) → {out_csv}")
else:
    print("\n⚠️ No valid data found for any year.")
