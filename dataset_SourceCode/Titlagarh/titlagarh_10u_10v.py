import cdsapi
import xarray as xr
import pandas as pd
import os
import time

# --- Initialize Copernicus CDS Client ---
client = cdsapi.Client(
    url='https://cds.climate.copernicus.eu/api',
    key='18f04497-8219-49a6-aefe-4826e9907662'
)

# --- Setup ---
years = [str(y) for y in range(1979, 2025)]  # 1979–2024 inclusive
variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind"
]

# Output folder
output_folder = "titlagarh_surface_daily_csv"
os.makedirs(output_folder, exist_ok=True)

def wait_for_file(file_path, retries=15, delay=4):
    """Wait until the file is fully closed and readable."""
    for attempt in range(retries):
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb"):
                    return True
            except PermissionError:
                print(f"⏳ Waiting for file lock to release ({attempt+1}/{retries})...")
                time.sleep(delay)
        else:
            time.sleep(2)
    return False

all_daily_files = []

for year in years:
    print(f"\n📅 Downloading ERA5 Surface Wind Data for Titlagarh - {year}...")
    output_nc_hourly = f"titlagarh_surface_6hourly_{year}.nc"
    output_nc_daily  = f"titlagarh_surface_daily_{year}.nc"
    output_csv_daily = os.path.join(output_folder, f"titlagarh_surface_daily_{year}.csv")

    # --- Titlagarh bounding box [North, West, South, East]
    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [year],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [20.8, 82.6, 19.8, 83.6],  # Titlagarh region
    }

    # --- Step 1: Download safely ---
    try:
        client.retrieve("reanalysis-era5-single-levels", request).download(output_nc_hourly)
        print(f"✅ Download requested for {year}: {output_nc_hourly}")

        # Wait until file is available
        if not wait_for_file(output_nc_hourly):
            print(f"❌ File {output_nc_hourly} still locked after waiting. Skipping {year}.")
            continue
    except Exception as e:
        print(f"❌ Error downloading {year}: {e}")
        continue

    # --- Step 2: Open dataset safely ---
    try:
        ds = xr.open_dataset(output_nc_hourly, engine="netcdf4")
    except Exception:
        print(f"⚠️ {year}: Not standard NetCDF, trying cfgrib...")
        ds = xr.open_dataset(output_nc_hourly, engine="cfgrib")

    # --- Step 3: Rename 'valid_time' → 'time' if needed ---
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # --- Step 4: Spatial and dimension averaging ---
    if "latitude" in ds.dims and "longitude" in ds.dims:
        ds = ds.mean(dim=["latitude", "longitude"])
    if "number" in ds.dims:
        ds = ds.mean(dim="number")
    if "expver" in ds.dims:
        ds = ds.mean(dim="expver")

    # --- Step 5: Convert to daily average ---
    print(f"🔄 Converting {year} to daily averages...")
    ds_daily = ds.resample(time="1D").mean()
    ds_daily.to_netcdf(output_nc_daily)
    print(f"✅ Saved daily NetCDF for {year}: {output_nc_daily}")

    # --- Step 6: Convert to CSV ---
    df = ds_daily.to_dataframe().reset_index()

    # Keep U and V components
    if "u10" in df.columns and "v10" in df.columns:
        df = df[["time", "u10", "v10"]]
        df.rename(columns={"u10": "u_component", "v10": "v_component"}, inplace=True)
    else:
        print(f"⚠️ {year}: Missing u10/v10 columns, skipping.")
        continue

    df.to_csv(output_csv_daily, index=False, float_format="%.4f")
    print(f"✅ Saved daily CSV for {year}: {output_csv_daily}")

    all_daily_files.append(output_csv_daily)

# --- Step 7: Merge all yearly CSVs ---
print("\n🔗 Merging all yearly CSVs into one master CSV...")
df_all = pd.concat([pd.read_csv(f) for f in all_daily_files], ignore_index=True)
merged_csv_path = os.path.join(output_folder, "titlagarh_surface_daily_1979_2024.csv")
df_all.to_csv(merged_csv_path, index=False, float_format="%.4f")

print("\n🎉 All years processed successfully for Titlagarh!")
print(f"📁 Final folder: {os.path.abspath(output_folder)}")
print(f"📄 Combined CSV: {merged_csv_path}")
