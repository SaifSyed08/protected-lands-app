import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pydeck as pdk

st.set_page_config("Find Similar Protected Lands", layout="wide")
st.title("🛰️ Compare Your Location to Protected Lands")

# 1) Upload CSV
uploaded = st.file_uploader("Upload a CSV with latitude,longitude in the first row", type="csv")
if not uploaded:
    st.warning("Please upload a CSV to continue.")
    st.stop()

df_user = pd.read_csv(uploaded)
# Make sure your uploaded CSV has these headers:
# latitude, longitude
df_user.columns = [col.lower() for col in df_user.columns]  # normalize casing

if not {"latitude", "longitude"}.issubset(df_user.columns):
    st.error("Your CSV must include 'latitude' and 'longitude' columns.")
    st.stop()

lat, lon = df_user.loc[0, "latitude"], df_user.loc[0, "longitude"]


# 2) Fetch climate with error handling
@st.cache_data(show_spinner=False)
def fetch_climate(latitude: float, longitude: float):
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
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
def fetch_elevation(latitude: float, longitude: float):
    try:
        r = requests.get("https://api.open-meteo.com/v1/elevation", params={"latitude": latitude, "longitude": longitude}, timeout=30)
        data = r.json()
        elev = data.get("elevation")
        if isinstance(elev, list):
            return float(elev[0])
        return float(elev)
    except Exception as e:
        st.error(f"Elevation API error: {e}")
        return np.nan

# 4) Run fetches
avg_temp, avg_precip = fetch_climate(lat, lon)
elevation = fetch_elevation(lat, lon)
if np.isnan(avg_temp) or np.isnan(avg_precip) or np.isnan(elevation):
    st.stop()

st.markdown(
    f"**Your location’s 30 yr avg temp:** {avg_temp:.1f} °C  •  "
    f"**precip:** {avg_precip:.1f} mm  •  "
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

input_df = pd.DataFrame([[avg_temp, avg_precip, elevation]], columns=features)
input_scaled = scaler.transform(input_df)
dists, idxs = nn.kneighbors(input_scaled)

top3 = df_pl.iloc[idxs[0]].copy()
top3["distance"] = dists[0]

# 7) Show results
st.subheader("Top 3 Similar Protected Lands")
st.table(top3[["NAME", "lat", "lon", "annual_temp_c", "annual_precip_mm", "elevation_m", "AREA_KM2", "distance"]])

# 8) Visualize on map with radius = √area
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
