import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import json
import os

st.set_page_config(page_title="Jakarta Flood Warning", layout="wide")
st.title("JakartaFloodPredict")
st.markdown("**Sistem Deteksi Dini & Peta Risiko Banjir** *Powered by JakartaFloodPredict Model (Multiple Linear Regression)*""")

CITY_CENTERS = {
    "Jakarta Pusat":   {"lat": -6.1805, "lon": 106.8284},
    "Jakarta Utara":   {"lat": -6.1384, "lon": 106.8645},
    "Jakarta Barat":   {"lat": -6.1674, "lon": 106.7637},
    "Jakarta Selatan": {"lat": -6.2615, "lon": 106.8106},
    "Jakarta Timur":   {"lat": -6.2250, "lon": 106.9004},
    "All":             {"lat": -6.2000, "lon": 106.8456} 
}

@st.cache_data
def load_geojson():
    if os.path.exists("jakarta_geo.json"):
        with open("jakarta_geo.json", "r") as f: return json.load(f)
    elif os.path.exists("jakarta_fixed.json"):
        with open("jakarta_fixed.json", "r") as f: return json.load(f)
    return None

jakarta_geojson = load_geojson()

@st.cache_data
def load_data():
    try:
        file_path = "dataset_banjirJakarta.csv"
        if not os.path.exists(file_path):
            file_path = "dataset_banjirJakarta.csv" 
            
        df = pd.read_csv(file_path)
        if df.shape[1] < 2: df = pd.read_csv(file_path, sep=";")
        
        if 'ketinggian_air_min_cm' in df.columns:
            df['waterLevel_cm'] = (df['ketinggian_air_min_cm'] + df['ketinggian_air_max_cm']) / 2
        
        df.columns = df.columns.str.strip()
        df['kota_administrasi'] = df['kota_administrasi'].str.strip()
        
        cols_needed = ['kota_administrasi', 'kecamatan', 'kelurahan', 'rw', 
                       'waterLevel_cm', 'rainFall_mm', 'rainDuration_minutes', 
                       'bulan_kejadian', 'tgl_kejadian'] 
        
        available_cols = [c for c in cols_needed if c in df.columns]
        df = df[available_cols].dropna()

        month_map = {1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni", 
                     7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November", 12: "Desember"}
        
        if df['bulan_kejadian'].dtype in [np.int64, np.float64]:
            df['bulan_kejadian'] = df['bulan_kejadian'].map(month_map)
            
        ordered_months = list(month_map.values())
        df['bulan_kejadian'] = pd.Categorical(df['bulan_kejadian'], categories=ordered_months, ordered=True)
        
        if 'rainFall_mm' in df.columns:
            X = df[['rainFall_mm', 'rainDuration_minutes']]
            y = df['waterLevel_cm']
            model = LinearRegression()
            model.fit(X, y)
        else:
            model = None

        return model, df, ordered_months

    except Exception as e:
        st.error(f"Error data: {e}")
        return None, None, None

model, df, ORDERED_MONTHS = load_data()

if model is None: st.stop()

st.sidebar.header("Lokasi Saya")

list_kota = sorted(df['kota_administrasi'].unique())
pilih_kota = st.sidebar.selectbox("Kota:", list_kota)

df_kota = df[df['kota_administrasi'] == pilih_kota]
list_kec = sorted(df_kota['kecamatan'].unique())
pilih_kec = st.sidebar.selectbox("Kecamatan:", list_kec)

df_kec = df_kota[df_kota['kecamatan'] == pilih_kec]
list_kel = sorted(df_kec['kelurahan'].unique())
pilih_kel = st.sidebar.selectbox("Kelurahan:", list_kel)

df_kel = df_kec[df_kec['kelurahan'] == pilih_kel]
list_rw = sorted(df_kel['rw'].unique())
pilih_rw = st.sidebar.selectbox("RW:", list_rw)

avg_flood_rw = df_kel[df_kel['rw'] == pilih_rw]['waterLevel_cm'].mean()
if pd.isna(avg_flood_rw): avg_flood_rw = df_kel['waterLevel_cm'].mean()
if pd.isna(avg_flood_rw): avg_flood_rw = df_kota['waterLevel_cm'].mean()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader(f"Simulasi di {pilih_kel.title()}")
    st.caption(f"Lokasi: Kec. {pilih_kec.title()}, {pilih_kota}, RW {pilih_rw}")
    
    rain = st.number_input("Curah Hujan (mm)", 0.0, 500.0, 50.0)
    dur = st.number_input("Durasi (menit)", 0, 600, 60)
    bulan = st.selectbox("Bulan", ORDERED_MONTHS)
    
    if st.button("Analisa Risiko", type="primary"):
        pred_base = max(0, model.predict([[rain, dur]])[0])
        avg_jakarta = df['waterLevel_cm'].mean()
        factor = (avg_flood_rw / avg_jakarta) if avg_jakarta > 0 else 1.0
        factor = max(0.8, min(factor, 1.5))
        final_pred = pred_base * factor
        
        idx = ORDERED_MONTHS.index(bulan) + 1
        is_rainy = idx in [10, 11, 12, 1, 2, 3]
        status, color = "AMAN", "green"
        if final_pred > 150: status, color = "EVAKUASI!", "darkred"
        elif final_pred > 70: status, color = "BAHAYA", "red"
        elif final_pred > 20: status, color = ("SIAGA", "orange") if is_rainy else ("WASPADA", "yellow")
        
        st.divider()
        st.metric("Prediksi (Lokal)", f"{final_pred:.1f} cm", delta=f"{factor:.2f}x Faktor Wilayah")
        st.markdown(f"<div style='background-color:{color};padding:15px;border-radius:10px;text-align:center;color:white;font-weight:bold;font-size:20px;'>{status}</div>", unsafe_allow_html=True)

with col2:
    st.subheader("Peta Risiko")
    
    df_map_filter = df[
        (df['bulan_kejadian'] == bulan) & 
        (df['kota_administrasi'] == pilih_kota)
    ]
    
    center_coords = CITY_CENTERS.get(pilih_kota, CITY_CENTERS["All"])
    zoom_level = 11.2 
    
    if jakarta_geojson:
        map_data = df_map_filter.groupby('kota_administrasi', observed=False)['waterLevel_cm'].mean().reset_index()
        
        if not map_data.empty:
            fig = px.choropleth_mapbox(
                map_data,
                geojson=jakarta_geojson,
                locations='kota_administrasi',
                featureidkey="properties.name",
                color='waterLevel_cm',
                color_continuous_scale="Reds",
                range_color=(0, 100),
                opacity=0.6,
                mapbox_style="carto-positron",
                zoom=zoom_level,
                center=center_coords,
                title=f"Risiko: {pilih_kota} ({bulan})",
                
                labels={
                    'kota_administrasi': 'Wilayah', 
                    'waterLevel_cm': 'Tinggi Air (cm)'
                },
                hover_name='kota_administrasi',
                hover_data={
                    'kota_administrasi': False, 
                    'waterLevel_cm': ':.1f'
                }
            )
            fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Tidak ada data banjir untuk **{pilih_kota}** di bulan **{bulan}**.")
    else:
        st.error("Peta Gagal Dimuat! Pastikan file 'jakarta_geo.json' ada.")
