import xarray as xr
import pandas as pd
import numpy as np
import os
import re

# === Paths ===
input_folder = r"C:\project\dataset\SriGanganagar\rainfall"
output_folder = r"C:\project\dataset\SriGanganagar\rainfall_sri_ganganagar_csv"
os.makedirs(output_folder, exist_ok=True)

# === Sri Ganganagar Coordinates (°N, °E) ===
LAT, LON = 29.92, 73.88
LAT_BOX, LON_BOX = 0.25, 0.25  # ±0.25° buffer (≈ 25–30 km area)

# === Helper Function: Extract Year from Filename ===
def extract_year(filename):
    match = re.search(r"(19|20)\d{2}", filename)
    return match.group(0) if match else "Unknown"

# === Storage for yearly CSV paths ===
all_csvs = []

# === Loop through all NetCDF rainfall files ===
for file in sorted(os.listdir(input_folder)):
    if not file.endswith(".nc"):
        continue

    path = os.path.join(input_folder, file)
    print(f"\n📘 Processing {file}...")

    try:
        # --- Open dataset ---
        ds = xr.open_dataset(path)

        # --- Identify coordinates and variables dynamically ---
        lat_name = next((n for n in ds.coords if "lat" in n.lower()), None)
        lon_name = next((n for n in ds.coords if "lon" in n.lower()), None)
        time_name = next((n for n in ds.coords if "time" in n.lower()), None)
        rain_var = next(
            (v for v in ds.data_vars if any(k in v.lower() for k in ["rain", "rf", "precip"])),
            None
        )

        if not (lat_name and lon_name and time_name and rain_var):
            print(f"⚠️ Skipping {file}: Missing coordinate or rainfall variable.")
            print(f"Available coords: {list(ds.coords)} | Variables: {list(ds.data_vars)}")
            continue

        print(f"🌧 Found rainfall variable: {rain_var}")

        # --- Select region (assumes ascending latitude; works for most IMD/ERA5) ---
        ds_city = ds.sel(
            {
                lat_name: slice(LAT - LAT_BOX, LAT + LAT_BOX),
                lon_name: slice(LON - LON_BOX, LON + LON_BOX)
            }
        )

        # --- If empty region, try interpolation (nearest point) ---
        if ds_city[rain_var].count() == 0:
            print("⚠️ Region slice empty → using nearest grid point instead.")
            ds_city = ds.interp({lat_name: LAT, lon_name: LON}, method="nearest")

        # --- Extract rainfall variable and clean ---
        rf = ds_city[rain_var]
        rf = rf.where(np.isfinite(rf), np.nan)

        # --- Spatial average if multiple grid cells ---
        if lat_name in rf.dims and lon_name in rf.dims:
            rf = rf.mean(dim=[lat_name, lon_name], skipna=True)

        # --- Ensure units are mm/day ---
        units = str(rf.attrs.get("units", "")).lower()
        if not units:
            print("ℹ️ No 'units' attribute found; assuming mm/day.")
        elif "m" in units and not "mm" in units:
            rf = rf * 1000  # Convert meters → millimeters
        rf.attrs["units"] = "mm/day"

        # --- Convert to DataFrame ---
        df = rf.to_dataframe().reset_index()
        df.rename(columns={rain_var: "Rainfall_mm"}, inplace=True)
        df.rename(columns={time_name: "Date"}, inplace=True)

        # --- Format Date ---
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
        df["Date"] = df["Date"].dt.strftime("%d-%m-%Y")

        # --- Add Year ---
        year = extract_year(file)
        df["Year"] = int(year) if year != "Unknown" else None

        # --- Skip empty datasets ---
        if df["Rainfall_mm"].isna().all():
            print(f"⚠️ {file}: All rainfall values are NaN, skipping.")
            continue

        # --- Save yearly CSV ---
        out_csv = os.path.join(output_folder, f"sri_ganganagar_rainfall_{year}.csv")
        df.to_csv(out_csv, index=False, float_format="%.3f")
        all_csvs.append(out_csv)
        print(f"✅ Saved {out_csv} ({len(df)} days)")

    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

# === Merge all yearly CSVs ===
if all_csvs:
    print("\n🔗 Merging all yearly rainfall CSVs into one file...")

    # Combine all yearly CSVs
    df_all = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
    df_all["Date"] = pd.to_datetime(df_all["Date"], format="%d-%m-%Y", errors="coerce")
    df_all.sort_values("Date", inplace=True)
    df_all["Date"] = df_all["Date"].dt.strftime("%d-%m-%Y")

    merged_csv = os.path.join(output_folder, "sri_ganganagar_rainfall_1979_2024.csv")

    # --- Fix: Delete old file if open or exists ---
    if os.path.exists(merged_csv):
        try:
            os.remove(merged_csv)
            print("🧹 Existing merged file removed before saving new one.")
        except PermissionError:
            merged_csv = os.path.join(output_folder, "sri_ganganagar_rainfall_1979_2024_new.csv")
            print("⚠️ Existing file locked, saving as new copy instead.")

    # --- Save final merged dataset ---
    df_all.to_csv(merged_csv, index=False, float_format="%.3f")

    print("\n🎉 All years processed successfully for Sri Ganganagar!")
    print(f"📁 Output Folder: {os.path.abspath(output_folder)}")
    print(f"📄 Final CSV: {merged_csv}")
else:
    print("⚠️ No valid rainfall data processed.")
