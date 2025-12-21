import cdsapi
import xarray as xr
import pandas as pd
import os, time

client = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="18f04497-8219-49a6-aefe-4826e9907662"
)

years = [str(y) for y in range(1979, 2025)]
variables = ["10m_u_component_of_wind", "10m_v_component_of_wind"]

output_folder = r"D:\chandrapur_surface_wind"
os.makedirs(output_folder, exist_ok=True)

def wait_for_file(path):
    for _ in range(20):
        if os.path.exists(path):
            try:
                with open(path, "rb"):
                    return True
            except:
                time.sleep(3)
        time.sleep(2)
    return False

all_csvs = []

for year in years:
    print(f"\n📅 Wind Data {year} – Chandrapur")

    nc_file = os.path.join(output_folder, f"chandrapur_wind_6hr_{year}.nc")
    csv_out = os.path.join(output_folder, f"chandrapur_wind_daily_{year}.csv")

    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [year],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": ["00:00", "06:00", "12:00", "18:00"],
        "area": [20.5, 78.8, 19.5, 79.8],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    client.retrieve(
        "reanalysis-era5-single-levels", request
    ).download(nc_file)

    if not wait_for_file(nc_file):
        continue

    ds = xr.open_dataset(nc_file)
    ds = ds.mean(dim=["latitude", "longitude"])
    ds_daily = ds.resample(time="1D").mean()

    df = ds_daily.to_dataframe().reset_index()
    df.to_csv(csv_out, index=False)

    all_csvs.append(csv_out)

pd.concat([pd.read_csv(f) for f in all_csvs]).to_csv(
    os.path.join(output_folder, "chandrapur_wind_1979_2024.csv"),
    index=False
)

print("✅ Chandrapur wind data completed")
