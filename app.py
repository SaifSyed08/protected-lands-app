import pandas as pd, requests
import streamlit as st

uploaded = st.file_uploader("Upload CSV")
if uploaded:
    df_user = pd.read_csv(uploaded)
    lat, lon = df_user.iloc[0, :2]  # first two cells = lat, lon
params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": "1991-01-01",
    "end_date":   "2020-12-31",
    "models":    ["MRI_AGCM3_2_S","EC_Earth3P_HR"],
    "daily":     ["temperature_2m_mean","precipitation_sum"],
    "temperature_unit":"celsius",
    "precipitation_unit":"mm"
}
r = requests.get("https://climate-api.open-meteo.com/v1/climate", params=params).json()
temps = r["daily"]["temperature_2m_mean"]
prec = r["daily"]["precipitation_sum"]
avg_temp   = sum(temps) / len(temps)
avg_precip = sum(prec)  / len(prec)   # total over days → divide by days if you want per‑day mean
r = requests.get(
    "https://api.open-meteo.com/v1/elevation",
    params={"latitude":lat,"longitude":lon}
).json()
elevation = r["elevation"]
df_pl = pd.read_csv("protected_lands.csv")
# assume df_pl has columns: lat, lon, avg_temp, avg_precip, elevation
features = ["avg_temp","avg_precip","elevation"]
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
nn.fit(df_pl[features])
dists, idxs = nn.kneighbors([[avg_temp, avg_precip, elevation]])
top3 = df_pl.iloc[idxs[0]]
st.write("### Top 3 Similar Protected Lands")
st.table(top3.assign(distance=dists[0]))

# map: rename so Streamlit knows lat/lon
st.map(top3.rename(columns={"lat":"latitude","lon":"longitude"}))
