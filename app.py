import streamlit as st
import pandas as pd
import requests
from sklearn.neighbors import NearestNeighbors

st.set_page_config("Find Similar Protected Lands", layout="wide")
st.title("🛰️ Compare Your Location to Protected Lands")

# 1) Upload & parse user CSV
uploaded = st.file_uploader("Upload a CSV with latitude,longitude in the first row", type="csv")
if not uploaded:
    st.warning("Please upload a CSV to continue.")
    st.stop()

df_user = pd.read_csv(uploaded)
lat, lon = df_user.iloc[0, 0], df_user.iloc[0, 1]

# 2) Cached fetch for 30‑yr climate normals
@st.cache_data(show_spinner=False)
def fetch_climate(latitude: float, longitude: float):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "1991-01-01",
        "end_date":   "2020-12-31",
        "daily":     ["temperature_2m_mean", "precipitation_sum"],
        "temperature_unit": "celsius",
        "precipitation_unit": "mm"
    }
    r = requests.get("https://climate-api.open-meteo.com/v1/climate", params=params).json()
    temps = r["daily"]["temperature_2m_mean"]
    prec  = r["daily"]["precipitation_sum"]
    avg_t = sum(temps) / len(temps)
    avg_p = sum(prec) 
    return avg_t, avg_p

# 3) Cached fetch for elevation
@st.cache_data(show_spinner=False)
def fetch_elevation(latitude: float, longitude: float):
    r = requests.get(
        "https://api.open-meteo.com/v1/elevation",
        params={"latitude": latitude, "longitude": longitude}
    ).json()
    return r.get("elevation", None)

# … after your cached fetch_elevation call …
avg_temp, avg_precip = fetch_climate(lat, lon)
elevation = fetch_elevation(lat, lon)

# if elevation is accidentally a list, grab the first element
if isinstance(elevation, list):
    elevation = float(elevation[0])

st.markdown(
    f"**Your location’s 30 yr avg temp:** {avg_temp:.1f} °C  •  "
    f"**precip:** {avg_precip:.1f} mm  •  "
    f"**elevation:** {elevation:.0f} m"
)

# 5) Load protected‑lands data
df_pl = pd.read_csv("protected_lands.csv")  # expects columns: lat, lon, avg_temp, avg_precip, elevation
features = ["annual_temp_c", "annual_precip_mm", "elevation_m"]# 5) Load & clean protected‑lands data
df_pl = pd.read_csv("protected_lands.csv")
features = ["annual_temp_c", "annual_precip_mm", "elevation_m"]

#from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 6) Drop NaNs
df_pl = df_pl.dropna(subset=features)

# 7) Scale the feature space (Z‑score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pl[features])

# 8) Fit neighbors on the scaled data
nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
nn.fit(X_scaled)

input_df = pd.DataFrame([[avg_temp, avg_precip, elevation]], columns=features)
input_scaled = scaler.transform(input_df)
dists, idxs = nn.kneighbors(input_scaled)

top3 = df_pl.iloc[idxs[0]].copy()
top3["distance"] = dists[0]

# 8) Display the table (using the new names)
st.subheader("Top 3 Similar Protected Lands")
st.table(top3[["NAME", "lat", "lon", "annual_temp_c", "annual_precip_mm", "elevation_m", "AREA_KM2", "distance"]])

import pydeck as pdk

map_df = top3.rename(columns={"lat": "latitude", "lon": "longitude"})

# Add a radius column using sqrt of AREA_KM2 * 1000 for visibility
map_df["radius"] = map_df["AREA_KM2"].apply(lambda x: (x**0.5) * 1000)

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

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{NAME}\n{AREA_KM2} km²"}))
