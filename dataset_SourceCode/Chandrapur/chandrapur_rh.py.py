import cdsapi
import xarray as xr
import pandas as pd
import os

client = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="18f04497-8219-49a6-aefe-4826e9907662"
)

output_folder = r"D:\chandrapur_rh"
os.makedirs(output_folder, exist_ok=True)

client.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "variable": "relative_humidity",
        "pressure_level": "1000",
        "year": [str(y) for y in range(1979, 2025)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": "00:00",
        "area": [20.5, 78.8, 19.5, 79.8],
        "data_format": "netcdf"
    }
).download(os.path.join(output_folder, "chandrapur_rh.nc"))

df = xr.open_dataset(
    os.path.join(output_folder, "chandrapur_rh.nc")
).to_dataframe().reset_index()

df.to_csv(
    os.path.join(output_folder, "chandrapur_rh_1979_2024.csv"),
    index=False
)

print("✅ Chandrapur RH completed")
