import cdsapi
import xarray as xr
import pandas as pd
import os
import time

# --- Initialize CDS API Client ---
client = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="18f04497-8219-49a6-aefe-4826e9907662"
)

# --- Configuration ---
years = [str(y) for y in range(1979, 2025)]  # 1979–2024 inclusive
variables = {
    "msl": "mean_sea_level_pressure",
    "msnlwrf": "mean_surface_net_long_wave_radiation_flux",
    "ssrd": "surface_solar_radiation_downwards"
}

# --- Nested output folder: .../surface_daily_msl_ssr_mslf/sri_ganganagar ---
base_folder = os.path.join("surface_daily_msl_ssr_mslf", "sri_ganganagar")
os.makedirs(base_folder, exist_ok=True)

def wait_for_file(path, timeout=60):
    """Wait until file is fully available."""
    for _ in range(timeout // 3):
        if os.path.exists(path) and os.path.getsize(path) > 10_000:
            try:
                with open(path, "rb"):
                    return True
            except PermissionError:
                pass
        time.sleep(3)
    return False

def safe_open_dataset(path):
    """Try opening with NetCDF, fallback to cfgrib."""
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except Exception:
        try:
            return xr.open_dataset(path, engine="cfgrib")
        except Exception:
            return None

# --- Main Loop ---
all_daily_files = []

for year in years:
    print(f"\n📅 Processing year {year} for Sri Ganganagar...")
    dfs_year = []

    for short, varname in variables.items():
        filename_hourly = os.path.join(base_folder, f"{short}_{year}.nc")
        filename_daily  = os.path.join(base_folder, f"{short}_daily_{year}.nc")
        csv_out         = os.path.join(base_folder, f"{short}_daily_{year}.csv")

        # 🔹 Sri Ganganagar area: [N, W, S, E] around (29.9N, 73.9E)
        request = {
            "product_type": ["reanalysis"],
            "variable": [varname],
            "year": [year],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [30.4, 73.4, 29.4, 74.4],  # Sri Ganganagar bounding box
        }

        # --- Step 1: Download each variable ---
        for attempt in range(3):
            try:
                print(f"⬇️  Downloading {varname} for {year} (attempt {attempt+1})...")
                client.retrieve("reanalysis-era5-single-levels", request).download(filename_hourly)
                if wait_for_file(filename_hourly):
                    print(f"✅ Downloaded {os.path.basename(filename_hourly)}")
                    break
                else:
                    print("⚠️ File incomplete, retrying...")
            except Exception as e:
                print(f"❌ Error downloading {varname}: {e}")
                time.sleep(5)
        else:
            print(f"❌ Skipping {varname} for {year} due to repeated failures.")
            continue

        # --- Step 2: Open dataset safely ---
        ds = safe_open_dataset(filename_hourly)
        if ds is None:
            print(f"⚠️ Skipping {varname} for {year} (file unreadable).")
            continue

        # Rename time coordinate if needed
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})

        # --- Step 3: Average spatially and resample to daily ---
        if "latitude" in ds.dims and "longitude" in ds.dims:
            ds = ds.mean(dim=["latitude", "longitude"])
        if "number" in ds.dims:
            ds = ds.mean(dim="number")
        if "expver" in ds.dims:
            ds = ds.mean(dim="expver")

        ds_daily = ds.resample(time="1D").mean()

        # --- Step 4: Save daily NetCDF & CSV ---
        ds_daily.to_netcdf(filename_daily)
        df = ds_daily.to_dataframe().reset_index()
        df.to_csv(csv_out, index=False, float_format="%.4f")
        print(f"📊 Saved {varname} daily CSV for {year}: {csv_out}")
        dfs_year.append(df)

    # --- Step 5: Merge all 3 variables for this year ---
    if dfs_year:
        df_merged = dfs_year[0]
        for df in dfs_year[1:]:
            df_merged = pd.merge(df_merged, df, on="time", how="outer")
        yearly_csv = os.path.join(base_folder, f"sri_ganganagar_surface_daily_{year}.csv")
        df_merged.to_csv(yearly_csv, index=False, float_format="%.4f")
        all_daily_files.append(yearly_csv)
        print(f"✅ Combined all variables for {year}: {yearly_csv}")

# --- Step 6: Merge all years into one file ---
print("\n🔗 Combining all yearly CSVs...")
df_all = pd.concat([pd.read_csv(f) for f in all_daily_files], ignore_index=True)
merged_csv = os.path.join(base_folder, "sri_ganganagar_surface_daily_1979_2024.csv")
df_all.to_csv(merged_csv, index=False, float_format="%.4f")

print("\n🎉 All years processed successfully for Sri Ganganagar!")
print(f"📁 Final folder: {os.path.abspath(base_folder)}")
print(f"📄 Merged CSV: {merged_csv}")
