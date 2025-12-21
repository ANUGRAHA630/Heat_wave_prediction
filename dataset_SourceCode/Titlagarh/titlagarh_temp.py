import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

# ==============================
# 📁 PATH CONFIGURATION
# ==============================
input_folder = r"C:\project\dataset\titlagarh_trial2\maxtemp"     # folder containing GRD files
output_folder = r"C:\project\dataset\titlagarh_trial2\maxtemp_titlagarh_csv"
os.makedirs(output_folder, exist_ok=True)

# ==============================
# 🌍 TITLAGARH COORDINATES
# ==============================
CITY = "Titlagarh"
LAT, LON = 20.29, 83.15   # Titlagarh location
LAT_STEP, LON_STEP = 1.0, 1.0  # IMD grid resolution (1° × 1°)
LAT_START, LON_START = 6.5, 66.5  # IMD grid lower-left corner
NY, NX = 31, 31           # grid size: 6.5–36.5°N, 66.5–96.5°E

# Create grid coordinate arrays
lats = np.round(np.arange(LAT_START, LAT_START + NY * LAT_STEP, LAT_STEP), 2)
lons = np.round(np.arange(LON_START, LON_START + NX * LON_STEP, LON_STEP), 2)

# ==============================
# 🧮 FUNCTION DEFINITIONS
# ==============================
def extract_year(filename):
    """Extracts 4-digit year (19xx or 20xx) from filename."""
    match = re.search(r"(19|20)\d{2}", filename)
    return int(match.group(0)) if match else None

def read_grd_file(file_path, year):
    """Reads an IMD GRD file, reshapes, and extracts Titlagarh Tmax."""
    try:
        raw = np.fromfile(file_path, dtype=np.float32)
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

    grid_cells = NY * NX
    if raw.size % grid_cells != 0:
        print(f"⚠️ Skipping {file_path}: size mismatch ({raw.size}).")
        return None

    ndays = int(raw.size / grid_cells)
    data = raw.reshape(ndays, NY, NX)
    dates = [datetime(year, 1, 1) + timedelta(days=i) for i in range(ndays)]

    # Convert to xarray Dataset
    ds = xr.Dataset(
        {"Tmax": (["time", "lat", "lon"], data)},
        coords={"time": dates, "lat": lats, "lon": lons},
    )

    # Extract nearest grid cell to Titlagarh
    tmax_city = ds["Tmax"].sel(lat=LAT, lon=LON, method="nearest")

    # Convert to DataFrame
    df = tmax_city.to_dataframe().reset_index()
    df.rename(columns={"Tmax": "Tmax_C"}, inplace=True)
    df["City"] = CITY
    df["Year"] = year
    return df[["time", "Tmax_C", "Year", "City"]]

# ==============================
# 🚀 MAIN PROCESSING
# ==============================
print(f"\n📂 Reading IMD GRD Tmax data for {CITY} from {input_folder}")
all_years = []

for file in sorted(os.listdir(input_folder)):
    if not file.lower().endswith(".grd"):
        continue

    year = extract_year(file)
    if not year:
        print(f"⚠️ No year detected in filename: {file}")
        continue

    print(f"\n📘 Processing Year {year}: {file}")
    file_path = os.path.join(input_folder, file)
    df_year = read_grd_file(file_path, year)

    if df_year is not None and not df_year.empty:
        all_years.append(df_year)
        print(f"✅ Year {year}: {len(df_year)} daily records extracted.")
    else:
        print(f"⚠️ Year {year}: No valid data extracted.")

# ==============================
# 💾 MERGE AND SAVE OUTPUT
# ==============================
if all_years:
    df_all = pd.concat(all_years, ignore_index=True)
    df_all.rename(columns={"time": "Date"}, inplace=True)
    df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
    df_all.dropna(subset=["Date"], inplace=True)
    df_all["Date"] = df_all["Date"].dt.strftime("%d-%m-%Y")

    df_all = df_all[["Date", "Tmax_C", "Year", "City"]]

    output_csv = os.path.join(output_folder, f"{CITY}_Tmax_Daily_1979_2024.csv")

    # Handle permission / existing file safely
    if os.path.exists(output_csv):
        try:
            os.remove(output_csv)
        except PermissionError:
            output_csv = os.path.join(output_folder, f"{CITY}_Tmax_Daily_1979_2024_new.csv")

    df_all.to_csv(output_csv, index=False, float_format="%.2f")

    print(f"\n🎉 Successfully saved full dataset:")
    print(f"📄 {output_csv}")
    print(f"📊 Total Records: {len(df_all):,}")
else:
    print("\n⚠️ No valid GRD files processed for Titlagarh.")
