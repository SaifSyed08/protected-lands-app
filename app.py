import streamlit as st
import pandas as pd
import requests
import numpy as np
import datetime
from io import BytesIO
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pydeck as pdk

# Attempt to import boto3 for AWS S3 access
try:
    import boto3
except ImportError:
    boto3 = None

# Attempt to import h5py for HDF parsing
try:
    import h5py
except ImportError:
    h5py = None

st.set_page_config("Find Similar Protected Lands", layout="wide")
st.title("🛰️ Compare Your Location to Protected Lands")

# 1) Upload CSV
uploaded = st.file_uploader("Upload a CSV with columns latitude, longitude in the first row", type="csv")
if not uploaded:
    st.warning("Please upload a CSV to continue.")
    st.stop()

# Read user CSV
df_user = pd.read_csv(uploaded)

# Extract latitude/longitude from user CSV
latitude = float(df_user.iloc[0]["latitude"])
longitude = float(df_user.iloc[0]["longitude"])

# 2) Fetch climate with error handling
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
        data = r.json()
        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_mean", [])
        precs = daily.get("precipitation_sum", [])
        if not temps or not precs:
            return np.nan, np.nan
        return float(np.mean(temps)), float(np.sum(precs))
    except Exception as e:
        st.error(f"Climate API error: {e}")
        return np.nan, np.nan

# 3) Fetch elevation safely
@st.cache_data(show_spinner=False)
def fetch_elevation(lat_val: float, lon_val: float):
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": lat_val, "longitude": lon_val},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()
        elev = data.get("elevation")
        if isinstance(elev, list):
            return float(elev[0])
        return float(elev)
    except Exception as e:
        st.error(f"Elevation API error: {e}")
        return np.nan

# 4) Run fetches
avg_temp, avg_precip = fetch_climate(latitude, longitude)
elevation = fetch_elevation(latitude, longitude)
if np.isnan(avg_temp) or np.isnan(avg_precip) or np.isnan(elevation):
    st.stop()

# Display summary
st.markdown(
    f"**Your location’s 30 yr avg temp:** {avg_temp:.1f} °C  •  "
    f"**total precip:** {avg_precip:.0f} mm  •  "
    f"**elevation:** {elevation:.0f} m"
)

# 5) Load & clean protected lands
df_pl = pd.read_csv("protected_lands.csv")
features = ["annual_temp_c", "annual_precip_mm", "elevation_m"]
df_pl = df_pl.dropna(subset=features)

# 6) Normalize features and fit neighbors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pl[features])
nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
nn.fit(X_scaled)

# 7) Query nearest neighbors
input_df = pd.DataFrame([[avg_temp, avg_precip, elevation]], columns=features)
input_scaled = scaler.transform(input_df)
dists, idxs = nn.kneighbors(input_scaled)

# 8) Prepare top3 results
top3 = df_pl.iloc[idxs[0]].copy()
top3["distance"] = dists[0]

# 9) Display results
st.subheader("Top 3 Similar Protected Lands")
st.table(top3[["NAME", "lat", "lon"] + features + ["AREA_KM2", "distance"]])

# 10) Map visualization
map_df = top3.rename(columns={"lat": "latitude", "lon": "longitude"})
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
    pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{NAME}\n{AREA_KM2} km²"}
    )
)

# 11) HDF parsing helper
def parse_hdf_for_point(stream_body, lat_val, lon_val, metric):
    if h5py is None:
        st.error("h5py is not installed. Please add h5py to your requirements.")
        return np.nan
    try:
        data_bytes = stream_body.read()
        with h5py.File(BytesIO(data_bytes), 'r') as f:
            lat_ds = next((ds for ds in f.keys() if 'latitude' in ds.lower()), None)
            lon_ds = next((ds for ds in f.keys() if 'longitude' in ds.lower()), None)
            if not lat_ds or not lon_ds:
                return np.nan
            lat_arr = f[lat_ds][:]
            lon_arr = f[lon_ds][:]
            dist = (lat_arr - lat_val)**2 + (lon_arr - lon_val)**2
            flat_idx = np.argmin(dist)
            i, j = np.unravel_index(flat_idx, lat_arr.shape)
            ds_name = next((ds for ds in f.keys() if metric.lower() in ds.lower()), None)
            if not ds_name:
                return np.nan
            val = f[ds_name][i, j]
            if 'ndvi' in ds_name.lower() or 'evi' in ds_name.lower():
                return float(val) * 0.0001
            return float(val)
    except Exception:
        return np.nan

# 12) Time Series Comparison
st.header("Time Series Comparison")
metric = st.selectbox("Choose metric", ["NDVI"])

if boto3 is None:
    st.error("boto3 is not installed. Please add boto3 to your requirements.")
else:
    s3 = boto3.client("s3")

    @st.cache_data(show_spinner=False)
    def fetch_modis_time_series(lat_val, lon_val, metric):
        bucket = "modis-pds"
        current_year = datetime.datetime.now().year
        years = list(range(current_year - 19, current_year + 1))
        values = []
        for year in years:
            key = f"MOD13Q1/{year}.001.hdf"
            try:
                obj = s3.get_object(Bucket=bucket, Key=key)
                value = parse_hdf_for_point(obj['Body'], lat_val, lon_val, metric)
            except Exception:
                value = np.nan
            values.append(value)
        return pd.DataFrame({"year": years, metric: values})

    if st.button("Show time series"):
        st.info(f"Fetching {metric} time series...")
        df_user_ts = fetch_modis_time_series(latitude, longitude, metric)
        df_plot = df_user_ts.set_index("year").rename(columns={metric: "Your Location"})
        for _, row in top3.iterrows():
            df_pa = fetch_modis_time_series(row["lat"], row["lon"], metric)
            df_plot[row["NAME"]] = df_pa[metric].values
        st.subheader(f"{metric} Over Time")
        st.line_chart(df_plot)
