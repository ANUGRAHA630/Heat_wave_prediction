import cdsapi
import xarray as xr
import pandas as pd
import os
import time

# --- Initialize Copernicus CDS Client ---
# Ensure your ~/.cdsapirc file is set up or provide the key here
client = cdsapi.Client(
    url='https://cds.climate.copernicus.eu',
    key='18f04497-8219-49a6-aefe-4826e9907662'
)

# --- Setup ---
years = [str(y) for y in range(1979, 2025)]  # 1979–2024 inclusive

variables = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind"
]

output_folder = "umiam_surface_daily_csv"
os.makedirs(output_folder, exist_ok=True)


def wait_for_file(file_path, retries=15, delay=4):
    """Wait until the file is fully written and unlocked."""
    for attempt in range(retries):
        if os.path.exists(file_path):
            try:
                # Check if file is readable/unlocked
                with open(file_path, "rb"):
                    return True
            except PermissionError:
                print(f"⏳ Waiting for file lock ({attempt+1}/{retries})...")
                time.sleep(delay)
        else:
            time.sleep(2)
    return False


all_daily_files = []

# === Loop through all years ===
for year in years:
    print(f"\n📅 Downloading ERA5 Surface Wind Data for {year} (Umiam, Meghalaya)...")

    output_nc_hourly = f"umiam_surface_6hourly_{year}.nc"
    output_nc_daily = f"umiam_surface_daily_{year}.nc"
    output_csv_daily = os.path.join(
        output_folder, f"umiam_surface_daily_{year}.csv"
    )

    # === Umiam region bounding box ===
    # Center: 25.66°N, 91.90°E
    # Area format: [North, West, South, East]
    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [year],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",

        # Bounding box: ±0.5° around Umiam center
        "area": [26.16, 91.40, 25.16, 92.40],
    }

    # --- Step 1: Download safely ---
    try:
        client.retrieve(
            "reanalysis-era5-single-levels",
            request
        ).download(output_nc_hourly)

        print(f"✅ Downloaded: {output_nc_hourly}")

        if not wait_for_file(output_nc_hourly):
            print(f"❌ File incomplete or locked. Skipping {year}.")
            continue
    except Exception as e:
        print(f"❌ Error downloading {year}: {e}")
        continue

    # --- Step 2: Open dataset safely ---
    try:
        ds = xr.open_dataset(output_nc_hourly, engine="netcdf4")
    except Exception:
        print(f"⚠️ {year}: Trying GRIB reader (cfgrib)...")
        try:
            ds = xr.open_dataset(output_nc_hourly, engine="cfgrib")
        except Exception as e:
            print(f"❌ Failed to open dataset: {e}")
            continue

    # Coordinate standardisation
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # --- Step 3: Spatial mean and dimension reduction ---
    if "latitude" in ds.dims and "longitude" in ds.dims:
        ds = ds.mean(dim=["latitude", "longitude"])
    
    # Handle optional ERA5 dimensions
    for dim in ["number", "expver"]:
        if dim in ds.dims:
            ds = ds.mean(dim=dim)

    # --- Step 4: Convert to daily means ---
    print(f"🔄 Converting {year} to daily averages...")
    ds_daily = ds.resample(time="1D").mean()
    ds_daily.to_netcdf(output_nc_daily)

    # --- Step 5: CSV export ---
    df = ds_daily.to_dataframe().reset_index()

    # Column mapping (ERA5 uses u10/v10 for 10m wind)
    col_map = {"u10": "u_component", "v10": "v_component"}
    existing_cols = [c for c in col_map.keys() if c in df.columns]
    
    if existing_cols:
        df = df[["time"] + existing_cols]
        df.rename(columns=col_map, inplace=True)

    df.to_csv(output_csv_daily, index=False, float_format="%.4f")
    print(f"📊 Saved: {output_csv_daily}")

    all_daily_files.append(output_csv_daily)

    # Clean up temporary NetCDF files to save space
    if os.path.exists(output_nc_hourly):
        os.remove(output_nc_hourly)


# --- Step 6: Merge all CSVs into one master file ---
if all_daily_files:
    print("\n🔗 Combining all yearly CSVs into one master file...")

    df_all = pd.concat(
        [pd.read_csv(f) for f in all_daily_files],
        ignore_index=True
    )

    merged_csv_path = os.path.join(
        output_folder, "umiam_surface_daily_1979_2024.csv"
    )

    # Final sort by time
    df_all["time"] = pd.to_datetime(df_all["time"])
    df_all.sort_values("time", inplace=True)
    df_all.to_csv(merged_csv_path, index=False, float_format="%.4f")

    print(f"\n🎉 All years processed successfully for Umiam!")
    print(f"📁 Output folder: {os.path.abspath(output_folder)}")
    print(f"📄 Final merged CSV: {merged_csv_path}")
else:
    print("\n⚠️ No data was successfully processed.")
