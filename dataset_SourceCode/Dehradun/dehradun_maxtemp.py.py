import os, re
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

BASE = r"C:\project\dataset\Dehradun"
input_folder = os.path.join(BASE, "maxtemp")
output_folder = os.path.join(BASE, "maxtemp_processed")
os.makedirs(output_folder, exist_ok=True)

CITY = "Dehradun"
LAT, LON = 30.32, 78.03

LAT_START, LON_START = 6.5, 66.5
NY, NX = 31, 31

lats = np.arange(LAT_START, LAT_START + NY, 1.0)
lons = np.arange(LON_START, LON_START + NX, 1.0)

def extract_year(name):
    m = re.search(r"(19|20)\d{2}", name)
    return int(m.group()) if m else None

all_years = []

for file in os.listdir(input_folder):
    if not file.endswith(".grd"):
        continue

    year = extract_year(file)
    raw = np.fromfile(os.path.join(input_folder, file), dtype=np.float32)
    ndays = raw.size // (NY * NX)

    data = raw.reshape(ndays, NY, NX)
    dates = [datetime(year, 1, 1) + timedelta(days=i) for i in range(ndays)]

    ds = xr.Dataset(
        {"Tmax": (["time", "lat", "lon"], data)},
        coords={"time": dates, "lat": lats, "lon": lons}
    )

    tmax_city = ds["Tmax"].sel(lat=LAT, lon=LON, method="nearest")
    df = tmax_city.to_dataframe().reset_index()

    df["Date"] = pd.to_datetime(df["time"]).dt.strftime("%d-%m-%Y")
    df["City"] = CITY

    out_csv = os.path.join(output_folder, f"dehradun_tmax_{year}.csv")
    df[["Date", "Tmax", "City"]].to_csv(out_csv, index=False)
    all_years.append(df)

pd.concat(all_years).to_csv(
    os.path.join(output_folder, "dehradun_tmax_1979_2024.csv"),
    index=False
)

print("✅ Dehradun Tmax completed")
