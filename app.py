import streamlit as st
import pandas as pd
import requests
import numpy as np
import datetime
import os
import json
import tempfile
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pydeck as pdk
import ee

# Initialize Earth Engine only if not already initialized
# Initialize Earth Engine safely (compatible with all versions)
try:
    ee.Initialize()
except Exception:
    service_account = st.secrets["GEE_SERVICE_ACCOUNT"]
    key_dict = json.loads(st.secrets["GEE_PRIVATE_KEY"])
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_file:
        json.dump(key_dict, tmp_file)
        tmp_file.flush()
        credentials = ee.ServiceAccountCredentials(service_account, tmp_file.name)
        ee.Initialize(credentials)

# Page configuration
st.set_page_config("Find Similar Protected Lands", layout="wide")
st.title("🛰️ Compare Your Location to Protected Lands")

# CSV upload
uploaded = st.file_uploader("Upload a CSV with columns latitude, longitude in the first row", type="csv")
if not uploaded:
    st.warning("Please upload a CSV to continue.")
    st.stop()

df_user = pd.read_csv(uploaded)
latitude = float(df_user.iloc[0]["latitude"])
longitude = float(df_user.iloc[0]["longitude"])

# Fetch 30-year climate averages
@st.cache_data(show_spinner=False)
def fetch_climate(lat_val: float, lon_val: float):
    params = {
        "latitude": lat_val,
        "longitude": lon_val,
        "start_date": "1991-01-01",
        "end_date": "2020-12-31",
        "daily": "temperature_2m_mean,precipitation_sum",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm"
    }
    try:
        r = requests.get("https://climate-api.open-meteo.com/v1/climate", params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("daily", {})
        temps = data.get("temperature_2m_mean", [])
        precs = data.get("precipitation_sum", [])
        return float(np.mean(temps)), float(np.sum(precs))
    except Exception as e:
        st.error(f"Climate API error: {e}")
        return np.nan, np.nan

# Fetch elevation
@st.cache_data(show_spinner=False)
def fetch_elevation(lat_val: float, lon_val: float):
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": lat_val, "longitude": lon_val},
            timeout=30
        )
        r.raise_for_status()
        elev = r.json().get("elevation")
        return float(elev[0] if isinstance(elev, list) else elev)
    except Exception as e:
        st.error(f"Elevation API error: {e}")
        return np.nan

def fetch_ee_ndvi_et_series(lat_val, lon_val):
    point = ee.Geometry.Point([lon_val, lat_val])
    years = list(range(2002, 2023))
    ndvi_vals = []
    et_vals = []

    for year in years:
        start, end = f"{year}-06-01", f"{year}-06-30"
        # build composites:
        ndvi_img = (ee.ImageCollection("MODIS/006/MOD13Q1")
                    .filterDate(start, end)
                    .select("NDVI")
                    .mean())
        et_img   = (ee.ImageCollection("MODIS/006/MOD16A2")
                    .filterDate(start, end)
                    .select("ET")
                    .mean())

        # pull the raw reduceRegion dicts:
        ndvi_dict = ndvi_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=3000,
            maxPixels=1e9
        ).getInfo()
        et_dict   = et_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=3000,
            maxPixels=1e9
        ).getInfo()

        # safely extract (or np.nan)
        raw_ndvi = ndvi_dict.get("NDVI")
        raw_et   = et_dict.get("ET")
        ndvi_vals.append(raw_ndvi   / 10000.0 if raw_ndvi is not None else np.nan)
        et_vals.append( raw_et     /    8.0 if raw_et   is not None else np.nan)

    return pd.DataFrame({"year": years, "NDVI": ndvi_vals, "ET": et_vals})

@st.cache_data
def fetch_ee_lst_gpp_series(lat_val: float, lon_val: float):
    point     = ee.Geometry.Point([lon_val, lat_val])
    years     = list(range(2002, 2023))
    lst_elems = []
    gpp_elems = []

    for year in years:
        start, end = f"{year}-06-01", f"{year}-06-30"

        lst_dict = (
            ee.ImageCollection("MODIS/006/MOD11A2")
              .filterDate(start, end)
              .select("LST_Day_1km")
              .mean()
              .reduceRegion(
                  reducer=ee.Reducer.mean(),
                  geometry=point,
                  scale=3000,         # named
                  maxPixels=1e9       # named
              )
        )

        gpp_dict = (
            ee.ImageCollection("MODIS/006/MOD17A2H")
              .filterDate(start, end)
              .select("Gpp")
              .mean()
              .reduceRegion(
                  reducer=ee.Reducer.mean(),
                  geometry=point,
                  scale=3000,          # named
                  maxPixels=1e9       # named
              )
        )

        safe_lst = ee.Algorithms.If(
            ee.Dictionary(lst_dict).contains("LST_Day_1km"),
            ee.Dictionary(lst_dict)
              .getNumber("LST_Day_1km")
              .multiply(0.02)
              .subtract(273.15),
            ee.Number(0)
        )
        safe_gpp = ee.Algorithms.If(
            ee.Dictionary(gpp_dict).contains("Gpp"),
            ee.Dictionary(gpp_dict).getNumber("Gpp"),
            ee.Number(0)
        )

        lst_elems.append(ee.Number(safe_lst))
        gpp_elems.append(ee.Number(safe_gpp))

    lst_vals = ee.List(lst_elems).getInfo()
    gpp_vals = ee.List(gpp_elems).getInfo()

    df = pd.DataFrame({
        "year": years,
        "LST_C (°C)": lst_vals,
        "GPP_gCm2_8day": gpp_vals
    })
    df.replace(0, np.nan, inplace=True)
    return df





# Compute and display local stats
avg_temp, avg_precip = fetch_climate(latitude, longitude)
elevation = fetch_elevation(latitude, longitude)
if np.isnan(avg_temp) or np.isnan(avg_precip) or np.isnan(elevation):
    st.stop()

st.markdown(
    f"**Your location’s 30 yr avg temp:** {avg_temp:.1f} °C  •  "
    f"**total precip:** {avg_precip:.0f} mm  •  "
    f"**elevation:** {elevation:.0f} m"
)

# Load protected lands and find nearest neighbor
df_pl = pd.read_csv("protected_lands.csv")
features = ["annual_temp_c", "annual_precip_mm", "elevation_m"]
df_pl = df_pl.dropna(subset=features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pl[features])
nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
nn.fit(X_scaled)

input_scaled = scaler.transform(
    pd.DataFrame([[avg_temp, avg_precip, elevation]], columns=features)
)
dist, idx = nn.kneighbors(input_scaled)

# Get best similar protected area
best = df_pl.iloc[idx[0][0]].copy()
best["distance"] = dist[0][0]

# Display best match
st.subheader("Best Similar Protected Land")
st.table(pd.DataFrame([best])[['NAME', 'lat', 'lon'] + features + ['AREA_KM2', 'distance']])


# Map visualization of the best one only
map_df = pd.DataFrame([best]).rename(columns={"lat": "latitude", "lon": "longitude"})
map_df["radius"] = map_df["AREA_KM2"].apply(lambda x: (x ** 0.5) * 1000)
layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position='[longitude, latitude]',
    get_radius='radius',
    get_fill_color='[0, 100, 255, 160]',
    pickable=True,
    auto_highlight=True
)
view_state = pdk.ViewState(
    latitude=map_df["latitude"].mean(),
    longitude=map_df["longitude"].mean(),
    zoom=4,
    pitch=0
)
st.pydeck_chart(
    pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{NAME}\n{AREA_KM2} km²"})
)

# Time Series Comparison using Earth Engine
st.header("Time Series Comparison")

# at the top of your Time Series section, initialize the flags
if "show_ndvi_et" not in st.session_state:
    st.session_state.show_ndvi_et = False
if "show_lst_gpp" not in st.session_state:
    st.session_state.show_lst_gpp = False

# place your two buttons side by side
col1, col2 = st.columns(2)
with col1:
    if st.button("Show NDVI & ET Time Series"):
        st.session_state.show_ndvi_et = True
with col2:
    if st.button("Show LST & GPP Time Series"):
        st.session_state.show_lst_gpp = True

# now render each chart block only if its flag is set
if st.session_state.show_ndvi_et:
    st.header("NDVI & ET Time Series")
    with st.spinner("Fetching data from Google Earth Engine…"):
        df_user_ts = fetch_ee_ndvi_et_series(latitude, longitude)
        df_best_ts = fetch_ee_ndvi_et_series(best["lat"], best["lon"])
        df_user_ts = df_user_ts.set_index("year")\
            .rename(columns={"NDVI": "Your NDVI", "ET": "Your ET"})
        df_best_ts = df_best_ts.set_index("year")\
            .rename(columns={"NDVI": f"{best['NAME']} NDVI", "ET": f"{best['NAME']} ET"})
        df = pd.concat([df_user_ts, df_best_ts], axis=1)

    st.subheader("NDVI Over Time")
    st.markdown("*NDVI (Normalized Difference Vegetation Index) measures green vegetation. Values range from -1 to 1, and higher values generally mean denser, healthier plant life.*")
    st.line_chart(df[[c for c in df.columns if "NDVI" in c]])
    st.subheader("ET Over Time")
    st.markdown("*ET (Evapotranspiration) reflects water loss from soil and plants. Higher ET can indicate more active vegetation and moisture, but also higher water demand.*")
    st.line_chart(df[[c for c in df.columns if "ET" in c]])

    csv_ndvi_et = df.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download NDVI & ET Data as CSV",
        data=csv_ndvi_et,
        file_name="ndvi_et_timeseries.csv",
        mime="text/csv"
    )

if st.session_state.show_lst_gpp:
    st.header("LST & GPP Time Series")
    with st.spinner("Fetching data from Google Earth Engine…"):
        df_user2 = fetch_ee_lst_gpp_series(latitude, longitude)\
            .set_index("year")\
            .rename(columns={
                "LST_C (°C)": "Your LST (°C)",
                "GPP_gCm2_8day": "Your GPP"
            })
        df_best2 = fetch_ee_lst_gpp_series(best["lat"], best["lon"])\
            .set_index("year")\
            .rename(columns={
                "LST_C (°C)": f"{best['NAME']} LST (°C)",
                "GPP_gCm2_8day": f"{best['NAME']} GPP"
            })
        df2 = pd.concat([df_user2, df_best2], axis=1)

    st.subheader("Land Surface Temperature Over Time")
    st.markdown("*LST (Land Surface Temperature) measures how hot the land surface is. Lower temperatures often suggest better vegetation cover or less urban heat.*")
    st.line_chart(df2[[c for c in df2.columns if "LST" in c]])
    st.subheader("Gross Primary Productivity Over Time")
    st.markdown("*GPP (Gross Primary Productivity) estimates how much carbon plants absorb via photosynthesis. Higher GPP indicates more plant growth and ecosystem productivity.*")
    st.line_chart(df2[[c for c in df2.columns if "GPP" in c]])

    # LST/GPP download
    csv_lst_gpp = df2.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download LST & GPP Data as CSV",
        data=csv_lst_gpp,
        file_name="lst_gpp_timeseries.csv",
        mime="text/csv"
    )
