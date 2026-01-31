import cdsapi
import xarray as xr
import pandas as pd
import os

# --- Initialize Copernicus CDS Client ---
client = cdsapi.Client(
    url='https://cds.climate.copernicus.eu/api',
    key='18f04497-8219-49a6-aefe-4826e9907662'
)

# --- Define range of years (1979–2024) ---
years = [str(y) for y in range(1979, 2025)]

# --- Variable and pressure level ---
variable = ["relative_humidity"]
pressure_level = ["1000"]

# --- Output folder for Adilabad ---
base_folder = os.path.join("era5_rh_1000hpa", "adilabad")
os.makedirs(base_folder, exist_ok=True)

all_daily_files = []

for year in years:
    print(f"\n📅 Downloading ERA5 Relative Humidity for Adilabad – {year}...")

    output_nc_hourly = os.path.join(base_folder, f"adilabad_rh_6hourly_{year}.nc")
    output_nc_daily  = os.path.join(base_folder, f"adilabad_rh_daily_{year}.nc")
    output_csv_daily = os.path.join(base_folder, f"adilabad_rh_daily_{year}.csv")

    # --- CDS API Request ---
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

        # Adilabad bounding box (N, W, S, E)
        # Center: ~19.66°N, 78.53°E
        "area": [20.16, 78.03, 19.16, 79.03],
    }

    # --- Step 1: Download ---
    try:
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            request
        ).download(output_nc_hourly)

        print(f"✅ Download complete for {year}: {output_nc_hourly}")
    except Exception as e:
        print(f"❌ Error downloading {year}: {e}")
        continue

    # --- Step 2: Open dataset ---
    try:
        ds = xr.open_dataset(output_nc_hourly, engine="netcdf4")
    except Exception:
        ds = xr.open_dataset(output_nc_hourly)

    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # --- Step 3: Remove extra dimensions and spatial average ---
    if "number" in ds.dims:
        ds = ds.mean(dim="number")
    if "expver" in ds.dims:
        ds = ds.mean(dim="expver")
    if "latitude" in ds.dims and "longitude" in ds.dims:
        ds = ds.mean(dim=["latitude", "longitude"])

    # Select pressure level
    if "pressure_level" in ds.dims:
        ds = ds.sel(pressure_level=1000)

    # --- Step 4: Convert to daily averages ---
    print(f"🔄 Converting {year} to daily averages...")
    ds_daily = ds.resample(time="1D").mean()
    ds_daily.to_netcdf(output_nc_daily)

    # --- Step 5: Convert to CSV ---
    df = ds_daily.to_dataframe().reset_index()

    # ERA5 RH variable is typically 'r'
    if "r" in df.columns:
        df = df[["time", "r"]]
        df.rename(columns={"r": "relative_humidity"}, inplace=True)

    # Clip RH to physical range (0-100%)
    if "relative_humidity" in df.columns:
        df["relative_humidity"] = df["relative_humidity"].clip(0, 100)

    df.to_csv(output_csv_daily, index=False, float_format="%.4f")
    print(f"📊 Saved daily CSV: {output_csv_daily}")

    all_daily_files.append(output_csv_daily)

# --- Step 6: Merge all years ---
if all_daily_files:
    print("\n🔗 Merging all yearly CSVs...")

    df_all = pd.concat([pd.read_csv(f) for f in all_daily_files], ignore_index=True)
    df_all["time"] = pd.to_datetime(df_all["time"])
    df_all = df_all.sort_values("time").reset_index(drop=True)

    merged_csv = os.path.join(base_folder, "adilabad_rh_daily_1979_2024.csv")
    df_all.to_csv(merged_csv, index=False, float_format="%.4f")

    print("\n🎉 All years processed successfully for Adilabad!")
    print(f"📁 Folder: {os.path.abspath(base_folder)}")
    print(f"📄 Final CSV: {merged_csv}")
else:
    print("⚠️ No files to merge.")
