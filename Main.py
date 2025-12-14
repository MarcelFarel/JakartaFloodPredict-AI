import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="JakartaFloodPredict", layout="centered")

st.title("JakartaFloodPredict")
st.markdown("""
**Sistem Deteksi Dini & Prediksi Risiko Banjir** *Powered by JakartaFloodPredict Model (Multiple Linear Regression)*
""")

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

        return model, X_test, y_test, y_pred_eval_func(model, X_test) # Return helper data

    except FileNotFoundError:
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def y_pred_eval_func(model, X_test):
    return model.predict(X_test)

model, X_test, y_test, y_pred_eval = load_data_and_train_model()

if model is None:
    st.error("ERROR FATAL: File 'dataset_nov25.csv' tidak ditemukan.")
    st.stop()

def predict_flood_status(rain_input, duration_input):
    
    if rain_input <= 0 or duration_input <= 0:
        return 0.0, 0.0, "AMAN", "green", "Cuaca normal. Tidak ada risiko banjir."

    input_data = np.array([[rain_input, duration_input]])
    predicted_level_cm = model.predict(input_data)[0]
    
    if predicted_level_cm < 0: predicted_level_cm = 0.0
    
    kapasitas_max = 100 
    risk_percentage = (predicted_level_cm / kapasitas_max) * 100
    
    if risk_percentage > 100: risk_percentage = 100
    if risk_percentage < 0: risk_percentage = 0

    if risk_percentage > 70:
        status = "DARURAT (Banjir Besar)"
        color = "red"
        action = "⚠️ PERINGATAN: Evakuasi Segera ke tempat tinggi!"
    elif risk_percentage >= 20:
        status = "SIAGA (Potensi Luapan)"
        color = "orange"
        action = "Persiapan Pompa & Pantau Situasi Terkini."
    else:
        status = "AMAN"
        color = "green"
        action = "Kondisi terkendali. Pemantauan Normal."

    return predicted_level_cm, risk_percentage, status, color, action

st.divider()
st.subheader("Panel Input Simulasi")
st.info("Masukkan parameter cuaca untuk menjalankan model **JakartaFloodPredict**:")

col_in1, col_in2 = st.columns(2)

with col_in1:
    st.markdown("**1. Intensitas Hujan**")
    kondisi_hujan = st.selectbox(
        "Bagaimana kondisi hujan saat ini?",
        options=[
            "Tidak Hujan / Mendung", 
            "Gerimis / Hujan Ringan", 
            "Hujan Sedang", 
            "Hujan Deras", 
            "Badai / Hujan Ekstrem"
        ],
        index=2
    )
    
    if kondisi_hujan == "Tidak Hujan / Mendung": input_rain = 0.0
    elif kondisi_hujan == "Gerimis / Hujan Ringan": input_rain = 20.0
    elif kondisi_hujan == "Hujan Sedang": input_rain = 50.0
    elif kondisi_hujan == "Hujan Deras": input_rain = 150.0
    else: input_rain = 350.0 
    
    st.caption(f"Input Model: {input_rain} mm")

with col_in2:
    st.markdown("**2. Durasi Hujan**")
    input_dur = st.slider(
        "Estimasi lama hujan (menit)", 
        min_value=0, max_value=300, value=60, step=5
    )
    
    jam = input_dur // 60
    menit = input_dur % 60
    if jam > 0: st.caption(f"Durasi: {jam} Jam {menit} Menit")
    else: st.caption(f"Durasi: {menit} Menit")

if st.button("Jalankan Prediksi", type="primary"):
    
    pred_level, risk_pct, status, color, action = predict_flood_status(input_rain, input_dur)
    
    st.divider()
    st.subheader("Hasil Analisa JakartaFloodPredict")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimasi Tinggi Air", f"{pred_level:.2f} cm")
        st.metric("Tingkat Risiko", f"{risk_pct:.1f} %")
    with col2:
        st.markdown("**Status Wilayah:**")
        st.markdown(f"<div style='background-color:{color}; padding:15px; border-radius:5px; color:white; text-align:center; font-weight:bold; font-size:20px;'>{status}</div>", unsafe_allow_html=True)
        st.write("") 
        st.info(f"{action}")

    st.subheader("Visualisasi Risiko")
    fig_chart = go.Figure()
    fig_chart.add_trace(go.Bar(
        y=['Risiko'], x=[risk_pct], name='Risiko', orientation='h',
        marker=dict(color=color, line=dict(color='black', width=1)),
        text=f"{risk_pct:.1f}%", textposition='inside'
    ))
    fig_chart.add_vline(x=20, line_dash="dash", line_color="green", annotation_text="Aman")
    fig_chart.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="Darurat")
    
    fig_chart.update_layout(
        xaxis=dict(range=[0, 100], title='Persentase Risiko (%)'),
        height=180, margin=dict(l=20, r=20, t=20, b=20), showlegend=False
    )
    st.plotly_chart(fig_chart, use_container_width=True)

    st.divider()
    with st.expander("Detail Evaluasi Akurasi Model (Data Teknis)"):
        mae = mean_absolute_error(y_test, y_pred_eval)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_eval))
        r2 = r2_score(y_test, y_pred_eval)

        st.markdown("### 1. Metrik Statistik")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (Rata-rata Error)", f"{mae:.2f} cm", help="Semakin kecil semakin baik")
        c2.metric("RMSE (Error Kuadrat)", f"{rmse:.2f} cm", help="Semakin kecil semakin baik")
        c3.metric("R² Score (Akurasi)", f"{r2:.4f}", help="Mendekati 1.0 berarti sempurna")

        st.markdown("### 2. Grafik Sebaran Prediksi")
        st.write("Grafik ini menunjukkan seberapa dekat prediksi model (Titik Biru) dengan kenyataan. Jika titik-titik menempel pada **Garis Merah Putus-putus**, berarti prediksi sangat akurat.")
        
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=y_test, y=y_pred_eval, mode='markers', name='Data Testing',
            marker=dict(color='royalblue', size=8, opacity=0.6)
        ))
        
        min_val = min(y_test.min(), y_pred_eval.min())
        max_val = max(y_test.max(), y_pred_eval.max())
        fig_eval.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val], mode='lines',
            name='Garis Ideal', line=dict(color='red', dash='dash', width=2)
        ))

        fig_eval.update_layout(
            xaxis_title="Nilai Aktual", yaxis_title="Nilai Prediksi",
            template="plotly_white", height=350, margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_eval, use_container_width=True)
