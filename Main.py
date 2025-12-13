import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Prediksi Banjir Jakarta", layout="centered")

st.title("Sistem Prediksi Banjir Jakarta")
st.markdown("Aplikasi berbasis **Multiple Linear Regression** untuk memprediksi risiko banjir (Skala Drainase Lokal) berdasarkan data historis dan simulasi.")

@st.cache_data
def load_data_and_train_model():
    try:
        df = pd.read_csv("dataset_nov25.csv", decimal=',')

        required_cols = ['waterLevel_cm', 'rainFall_mm', 'rainDuration_minutes']
        if not all(col in df.columns for col in required_cols):
             st.error(f"Dataset tidak memiliki kolom yang sesuai: {required_cols}")
             st.stop()

        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)

        X = df[['rainFall_mm', 'rainDuration_minutes']]
        y = df['waterLevel_cm']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        return model, X_test, y_test

    except FileNotFoundError:
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

model, X_test, y_test = load_data_and_train_model()

if model is None:
    st.error("ERROR FATAL: File 'dataset_nov25.csv' tidak ditemukan.")
    st.stop()

def predict_flood_status(rain_input, duration_input):
    
    # Jika tidak ada hujan (0 mm) ATAU durasinya 0 menit, maka tidak mungkin banjir.
    if rain_input <= 0 or duration_input <= 0:
        predicted_level_cm = 0.0
        risk_percentage = 0.0
        status = "AMAN"
        color = "green"
        action = "Cuaca Cerah / Normal. Tidak ada risiko."
        return predicted_level_cm, risk_percentage, status, color, action

    input_data = np.array([[rain_input, duration_input]])
    predicted_level_cm = model.predict(input_data)[0]
    
    if predicted_level_cm < 0:
        predicted_level_cm = 0.0
    
    kapasitas_max = 100 
    risk_percentage = (predicted_level_cm / kapasitas_max) * 100
    
    if risk_percentage > 100: risk_percentage = 100
    if risk_percentage < 0: risk_percentage = 0

    if risk_percentage > 70:
        status = "DARURAT (Banjir Besar)"
        color = "red"
        action = "Evakuasi Segera ke tempat tinggi!"
    elif risk_percentage >= 20:
        status = "SIAGA (Potensi Luapan)"
        color = "orange"
        action = "Persiapan Pompa & Pantau Situasi."
    else:
        status = "AMAN"
        color = "green"
        action = "Pemantauan Normal."

    return predicted_level_cm, risk_percentage, status, color, action

st.divider()
st.subheader("Panel Input Simulasi")
st.info("Silakan masukkan kondisi cuaca yang teramati saat ini:")

col_in1, col_in2 = st.columns(2)

with col_in1:
    st.markdown("**1. Kondisi Hujan**")
    kondisi_hujan = st.selectbox(
        "Seberapa deras intensitas hujan?",
        options=[
            "Tidak Hujan / Mendung", 
            "Gerimis / Hujan Ringan", 
            "Hujan Sedang", 
            "Hujan Deras", 
            "Badai / Hujan Ekstrem"
        ],
        index=2 
    )
    
    if kondisi_hujan == "Tidak Hujan / Mendung":
        input_rain = 0.0
    elif kondisi_hujan == "Gerimis / Hujan Ringan":
        input_rain = 20.0
    elif kondisi_hujan == "Hujan Sedang":
        input_rain = 50.0
    elif kondisi_hujan == "Hujan Deras":
        input_rain = 150.0
    else: 
        input_rain = 350.0
    
    st.caption(f"Sistem menggunakan nilai intensitas: {input_rain} mm")

with col_in2:
    st.markdown("**2. Durasi Hujan**")
    input_dur = st.slider(
        "Perkiraan lama hujan (menit)", 
        min_value=0, 
        max_value=300, 
        value=60,
        step=5,
        help="Geser ke 0 jika hujan baru saja dimulai atau belum terjadi."
    )
    
    jam = input_dur // 60
    menit = input_dur % 60
    if jam > 0 and menit > 0:
        st.caption(f"Waktu terkonversi: {jam} Jam {menit} Menit")
    elif jam > 0:
        st.caption(f"Waktu terkonversi: {jam} Jam")
    else:
        st.caption(f"Waktu terkonversi: {menit} Menit")

if st.button("Jalankan Prediksi", type="primary"):
    
    pred_level, risk_pct, status, color, action = predict_flood_status(input_rain, input_dur)
    
    st.divider()
    st.subheader("Hasil Prediksi")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimasi Tinggi Air", f"{pred_level:.2f} cm")
        st.metric("Tingkat Risiko", f"{risk_pct:.1f} %")
    with col2:
        st.markdown("**Status Wilayah:**")
        st.markdown(f"<div style='background-color:{color}; padding:15px; border-radius:5px; color:white; text-align:center; font-weight:bold; font-size:20px;'>{status}</div>", unsafe_allow_html=True)
        st.write("") 
        st.info(f"Saran Tindakan: {action}")

    st.subheader("Grafik Risiko Banjir")
    fig_chart = go.Figure()
    
    fig_chart.add_trace(go.Bar(
        y=['Level Risiko'], 
        x=[risk_pct], 
        name='Persentase Risiko', 
        orientation='h',
        marker=dict(color=color, line=dict(color='black', width=1)),
        text=f"{risk_pct:.1f}%", 
        textposition='inside', 
        insidetextanchor='middle'
    ))
    
    fig_chart.add_vline(x=20, line_dash="dash", line_color="green", annotation_text="Batas Aman (20%)", annotation_position="top right")
    fig_chart.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="Batas Darurat (70%)", annotation_position="top right")
    
    fig_chart.update_layout(
        xaxis=dict(range=[0, 100], title='Persentase Kemungkinan Banjir (%)'),
        height=200, 
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False
    )
    st.plotly_chart(fig_chart, use_container_width=True)

with st.expander("Lihat Detail Evaluasi Model"):
    y_pred_eval = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_eval)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_eval))
    r2 = r2_score(y_test, y_pred_eval)

    st.write("Metrik performa model (Berdasarkan data testing):")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (Mean Absolute Error)", f"{mae:.2f} cm")
    c2.metric("RMSE (Root Mean Sq. Error)", f"{rmse:.2f} cm")
    c3.metric("R² Score (Akurasi)", f"{r2:.4f}")