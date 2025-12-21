import xarray as xr
import pandas as pd
import os, re

BASE = r"C:\project\dataset\Dehradun"
input_folder = os.path.join(BASE, "rainfall")
output_folder = os.path.join(BASE, "rainfall_processed")
os.makedirs(output_folder, exist_ok=True)

LAT, LON = 30.32, 78.03

def extract_year(name):
    m = re.search(r"(19|20)\d{2}", name)
    return m.group() if m else "unknown"

all_csvs = []

for file in os.listdir(input_folder):
    if not file.endswith(".nc"):
        continue

    ds = xr.open_dataset(os.path.join(input_folder, file))

    lat = next(c for c in ds.coords if "lat" in c.lower())
    lon = next(c for c in ds.coords if "lon" in c.lower())
    time = next(c for c in ds.coords if "time" in c.lower())
    rain = next(v for v in ds.data_vars if "rain" in v.lower() or "precip" in v.lower())

    rf = ds[rain].interp({lat: LAT, lon: LON})
    rf_daily = rf.resample({time: "1D"}).sum()

    df = rf_daily.to_dataframe().reset_index()
    df.rename(columns={rain: "Rain_mm"}, inplace=True)

    out = os.path.join(output_folder, f"dehradun_rain_{extract_year(file)}.csv")
    df.to_csv(out, index=False)
    all_csvs.append(out)

pd.concat([pd.read_csv(f) for f in all_csvs]).to_csv(
    os.path.join(output_folder, "dehradun_rain_1979_2024.csv"),
    index=False
)

print("✅ Dehradun rainfall completed")
