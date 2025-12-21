import cdsapi
import xarray as xr
import pandas as pd
import os

client = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="18f04497-8219-49a6-aefe-4826e9907662"
)

years = [str(y) for y in range(1979, 2025)]
variables = {
    "msl": "mean_sea_level_pressure",
    "ssrd": "surface_solar_radiation_downwards",
    "msnlwrf": "mean_surface_net_long_wave_radiation_flux"
}

output_folder = r"D:\chandrapur_surface_flux"
os.makedirs(output_folder, exist_ok=True)

all_csvs = []

for year in years:
    dfs = []

    for short, var in variables.items():
        nc = os.path.join(output_folder, f"chandrapur_{short}_{year}.nc")

        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": var,
                "year": year,
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00", "06:00", "12:00", "18:00"],
                "area": [20.5, 78.8, 19.5, 79.8],
                "data_format": "netcdf"
            }
        ).download(nc)

        ds = xr.open_dataset(nc)
        ds = ds.mean(dim=["latitude", "longitude"])
        df = ds.resample(time="1D").mean().to_dataframe().reset_index()
        dfs.append(df)

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="time")

    csv = os.path.join(output_folder, f"chandrapur_surface_{year}.csv")
    merged.to_csv(csv, index=False)
    all_csvs.append(csv)

pd.concat([pd.read_csv(f) for f in all_csvs]).to_csv(
    os.path.join(output_folder, "chandrapur_surface_1979_2024.csv"),
    index=False
)

print("✅ Chandrapur surface variables completed")
