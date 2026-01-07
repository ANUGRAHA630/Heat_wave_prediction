import xarray as xr
import pandas as pd
import glob
import os

# === USER SETTINGS ===
DATA_FOLDER = r"C:\Users\tjpri\Downloads\dataset\rainfall\rainfall\rainfall"
OUTPUT_FILE = "panaji_daily_avg_rainfall_1979_2024.csv"

# Panaji bounding box (~15.49°N, 73.83°E)
LAT_MIN, LAT_MAX = 15.0, 16.0
LON_MIN, LON_MAX = 73.3, 74.3

# === LOAD ALL .nc FILES ===
nc_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.nc")))
if not nc_files:
    raise FileNotFoundError("❌ No NetCDF files found in the folder!")

print(f"📁 Found {len(nc_files)} NetCDF files.\n")

all_data = []

for nc_path in nc_files:
    print(f"📄 Processing {os.path.basename(nc_path)} ...")
    ds = xr.open_dataset(nc_path)

    # --- Detect LAT / LON / TIME ---
    lat_name = [c for c in ds.coords if "lat" in c.lower()][0]
    lon_name = [c for c in ds.coords if "lon" in c.lower()][0]
    time_name = [c for c in ds.coords if "time" in c.lower()][0]

    # --- Detect rainfall variable ---
    rain_candidates = [
        v for v in ds.data_vars
        if "rain" in v.lower()
        or "rf" in v.lower()
        or "precip" in v.lower()
        or v.lower() in ["pr", "tp"]
    ]

    if not rain_candidates:
        print("⚠️ No rainfall variable found, skipping file.")
        continue

    rain_var = rain_candidates[0]

    # --- Subset Panaji region ---
    ds_panaji = ds.sel(
        {
            lat_name: slice(LAT_MIN, LAT_MAX),
            lon_name: slice(LON_MIN, LON_MAX)
        }
    )

    if ds_panaji[rain_var].size == 0:
        print("⚠️ Region slice empty, skipping file.")
        continue

    # --- Convert to DataFrame ---
    df = ds_panaji[[rain_var]].to_dataframe().reset_index()

    df.rename(columns={
        time_name: "date",
        rain_var: "rainfall_mm"
    }, inplace=True)

    # --- Compute daily average rainfall ---
    daily_avg = (
        df.groupby("date")["rainfall_mm"]
        .mean()
        .reset_index()
    )

    all_data.append(daily_avg)

# === MERGE ALL YEARS ===
if not all_data:
    raise RuntimeError("❌ No valid rainfall data found for Panaji!")

combined_df = pd.concat(all_data, ignore_index=True)
combined_df.sort_values("date", inplace=True)

combined_df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Successfully saved:")
print("   ", OUTPUT_FILE)
print(f"📅 Total days processed: {combined_df.shape[0]}")
