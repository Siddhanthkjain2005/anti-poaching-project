import ee
import pandas as pd

# Initialize Earth Engine
ee.Initialize()

# INPUT / OUTPUT
INPUT_CSV = "data/anti_poaching.csv"
OUTPUT_CSV = "data/anti_poaching_with_env.csv"

# Load dataset
df = pd.read_csv(INPUT_CSV)

# Convert CSV rows to EE FeatureCollection
features = []
for _, row in df.iterrows():
    point = ee.Geometry.Point([row["longitude"], row["latitude"]])
    feat = ee.Feature(point, row.to_dict())
    features.append(feat)

fc = ee.FeatureCollection(features)

# -----------------------------
# 1) NIGHT LIGHT (VIIRS)
# -----------------------------
viirs = (
    ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
    .filterDate("2020-01-01", "2022-12-31")
    .select("avg_rad")
    .median()
)

# -----------------------------
# 2) VEGETATION (NDVI - MODIS)
# -----------------------------
ndvi = (
    ee.ImageCollection("MODIS/006/MOD13Q1")
    .filterDate("2020-01-01", "2022-12-31")
    .select("NDVI")
    .median()
    .multiply(0.0001)  # scale factor
)

# -----------------------------
# 3) ELEVATION (SRTM)
# -----------------------------
elevation = ee.Image("USGS/SRTMGL1_003")

# Combine all bands
env_image = viirs.rename("nightlight") \
    .addBands(ndvi.rename("ndvi")) \
    .addBands(elevation.rename("elevation"))

# Sample environmental values at points
sampled = env_image.sampleRegions(
    collection=fc,
    scale=500,
    geometries=False
)

# Convert to Pandas
data = sampled.getInfo()["features"]

records = []
for f in data:
    props = f["properties"]
    records.append(props)

env_df = pd.DataFrame(records)

# Save enriched dataset
env_df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
