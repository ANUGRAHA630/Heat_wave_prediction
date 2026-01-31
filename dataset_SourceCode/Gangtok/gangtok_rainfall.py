import xarray as xr
import pandas as pd
import glob
import os

# === USER SETTINGS ===
DATA_FOLDER = r"C:\Users\tjpri\Downloads\dataset\rainfall\rainfall\rainfall"
OUTPUT_FILE = "gangtok_daily_avg_rainfall_1979_2024.csv"

# Gangtok bounding box (~27.33°N, 88.62°E)
LAT_MIN, LAT_MAX = 26.83, 27.83
LON_MIN, LON_MAX = 88.12, 89.12

# === LOAD ALL .nc FILES ===
nc_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*.nc")))
if not nc_files:
    raise FileNotFoundError("❌ No NetCDF files found in the folder!")

print(f"📁 Found {len(nc_files)} NetCDF files.\n")

all_data = []

for nc_path in nc_files:
    print(f"📄 Processing {os.path.basename(nc_path)} ...")
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        print(f"❌ Error opening {nc_path}: {e}")
        continue

    # --- Detect LAT / LON / TIME ---
    lat_names = [c for c in ds.coords if "lat" in c.lower()]
    lon_names = [c for c in ds.coords if "lon" in c.lower()]
    time_names = [c for c in ds.coords if "time" in c.lower()]

    if not lat_names or not lon_names or not time_names:
        print(f"⚠️ Missing coordinates in {os.path.basename(nc_path)}, skipping.")
        continue

    lat_name, lon_name, time_name = lat_names[0], lon_names[0], time_names[0]

    # --- Detect rainfall variable ---
    rain_candidates = [
        v for v in ds.data_vars
        if any(keyword in v.lower() for keyword in ["rain", "rf", "precip", "pr", "tp"])
    ]

    if not rain_candidates:
        print("⚠️ No rainfall variable found, skipping file.")
        continue

    rain_var = rain_candidates[0]

    # --- Subset Gangtok region ---
    # Check both slice directions because some IMD datasets use descending latitudes
    ds_gangtok = ds.sel({lat_name: slice(LAT_MIN, LAT_MAX), lon_name: slice(LON_MIN, LON_MAX)})

    if ds_gangtok[rain_var].size == 0:
        ds_gangtok = ds.sel({lat_name: slice(LAT_MAX, LAT_MIN), lon_name: slice(LON_MIN, LON_MAX)})

    if ds_gangtok[rain_var].size == 0:
        print("⚠️ Region slice empty for Gangtok, skipping file.")
        continue

    # --- Convert to DataFrame ---
    df = ds_gangtok[[rain_var]].to_dataframe().reset_index()

    df.rename(columns={
        time_name: "date",
        rain_var: "rainfall_mm"
    }, inplace=True)

    # --- Compute daily average rainfall ---
    # Ensure date is in datetime format before grouping
    df['date'] = pd.to_datetime(df['date'])
    daily_avg = df.groupby("date")["rainfall_mm"].mean().reset_index()
    all_data.append(daily_avg)

# === MERGE ALL YEARS ===
if not all_data:
    raise RuntimeError("❌ No valid rainfall data found for Gangtok!")

combined_df = pd.concat(all_data, ignore_index=True)
combined_df.sort_values("date", inplace=True)

# Standardize date format to YYYY-MM-DD
combined_df["date"] = combined_df["date"].dt.strftime('%Y-%m-%d')

combined_df.to_csv(OUTPUT_FILE, index=False)

print("\n✅ Successfully saved:")
print("   ", OUTPUT_FILE)
print(f"📅 Total days processed: {combined_df.shape[0]}")
