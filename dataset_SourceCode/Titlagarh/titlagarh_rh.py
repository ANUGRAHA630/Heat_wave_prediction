import cdsapi
import xarray as xr
import pandas as pd
import os

# --- Initialize Copernicus CDS Client ---
client = cdsapi.Client(
    url='https://cds.climate.copernicus.eu/api',
    key='18f04497-8219-49a6-aefe-4826e9907662'
)

# --- Define range of years ---
years = [str(y) for y in range(1979, 2026)]  # 1979–2025 inclusive

# --- Variable and Pressure level ---
variable = ["relative_humidity"]
pressure_level = ["1000"]

# --- Output folder ---
output_folder = "titlagarh_rh_daily_csv"
os.makedirs(output_folder, exist_ok=True)

# --- List to hold all daily CSV filenames ---
all_daily_files = []

for year in years:
    print(f"\n📅 Downloading ERA5 Relative Humidity for Titlagarh - {year}...")
    output_nc_hourly = os.path.join(output_folder, f"titlagarh_rh_6hourly_{year}.nc")
    output_nc_daily = os.path.join(output_folder, f"titlagarh_rh_daily_{year}.nc")
    output_csv_daily = os.path.join(output_folder, f"titlagarh_rh_daily_{year}.csv")

    # --- Define CDS API request ---
    request = {
        "product_type": ["reanalysis"],
        "variable": variable,
        "pressure_level": pressure_level,
        "year": [year],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",

        # 🔸 Titlagarh region (North, West, South, East)
        # Center: ~20.29°N, 83.15°E
        # Bounding box covering local surroundings
        "area": [20.8, 82.6, 19.8, 83.6],
    }

    # --- Step 1: Download the data for that year ---
    try:
        client.retrieve("reanalysis-era5-pressure-levels", request).download(output_nc_hourly)
        print(f"✅ Download complete for {year}: {output_nc_hourly}")
    except Exception as e:
        print(f"❌ Error downloading {year}: {e}")
        continue

    # --- Step 2: Open dataset and clean up ---
    ds = xr.open_dataset(output_nc_hourly)

    # Rename 'valid_time' → 'time' (for pressure-level datasets)
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # ✅ Average across spatial/extra dimensions
    if "number" in ds.dims:
        ds = ds.mean(dim="number")
    if "expver" in ds.dims:
        ds = ds.mean(dim="expver")
    if "latitude" in ds.dims and "longitude" in ds.dims:
        ds = ds.mean(dim=["latitude", "longitude"])
    if "pressure_level" in ds.dims:
        ds = ds.sel(pressure_level=1000)

    # --- Step 3: Convert 6-hourly → daily mean ---
    print(f"🔄 Converting {year} to daily averages...")
    ds_daily = ds.resample(time="1D").mean()
    ds_daily.to_netcdf(output_nc_daily)
    print(f"✅ Saved daily NetCDF for {year}: {output_nc_daily}")

    # --- Step 4: Convert to CSV ---
    print(f"📊 Converting {year} daily data to CSV...")
    df = ds_daily.to_dataframe().reset_index()

    # Keep only relevant columns
    if "r" in df.columns:
        df = df[["time", "r"]]
        df.rename(columns={"r": "relative_humidity"}, inplace=True)

    df.to_csv(output_csv_daily, index=False, float_format="%.4f")
    print(f"✅ Saved daily CSV for {year}: {output_csv_daily}")

    all_daily_files.append(output_csv_daily)

    # Optional cleanup (uncomment if needed)
    # os.remove(output_nc_hourly)

# --- Step 5: Merge all yearly CSVs into one file ---
print("\n🔗 Merging all yearly CSV files into one file...")
df_all = pd.concat([pd.read_csv(f) for f in all_daily_files], ignore_index=True)
merged_csv = os.path.join(output_folder, "titlagarh_rh_daily_1979_2025.csv")
df_all.to_csv(merged_csv, index=False, float_format="%.4f")

print("\n🎉 All years processed successfully for Titlagarh!")
print(f"📁 Final folder: {os.path.abspath(output_folder)}")
print(f"📄 Final CSV: {merged_csv}")
