import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Jakarta Air Quality & AI Forecasting",
    page_icon="🤖",
    layout="wide"
)

# --- 2. FUNGSI LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/df_clean.csv')
    except:
        st.error("File 'df_clean.csv' tidak ditemukan.")
        return pd.DataFrame()

    # Cleaning Dasar: Hapus Stasiun '0'
    df = df[df['station_name'] != '0'] 
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Cleaning Numerik
    numeric_cols = ['max', 'pm10', 'pm25', 'so2', 'co', 'o3', 'no2']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Atribut Waktu
    day_map = {0: 'Senin', 1: 'Selasa', 2: 'Rabu', 3: 'Kamis', 4: 'Jumat', 5: 'Sabtu', 6: 'Minggu'}
    df['day_name'] = df['Date'].dt.dayofweek.map(day_map)
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df['month_name'] = df['Date'].dt.month.map(month_map)
    
    # Ordering Categories
    df['day_name'] = pd.Categorical(df['day_name'], categories=['Senin','Selasa','Rabu','Kamis','Jumat','Sabtu','Minggu'], ordered=True)
    df['month_name'] = pd.Categorical(df['month_name'], categories=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ordered=True)
    
    # Order Category ISPU
    cat_order = ['BAIK', 'SEDANG', 'TIDAK SEHAT', 'SANGAT TIDAK SEHAT', 'BERBAHAYA']
    df['category'] = pd.Categorical(df['category'], categories=cat_order, ordered=True)

    return df

@st.cache_resource
def load_model_for_station(station_name):
    """
    Memuat model spesifik berdasarkan nama stasiun yang dipilih user.
    """
    # Ubah nama stasiun jadi format filename (misal: "Kebon Jeruk" -> "model_kebon_jeruk.pkl")
    safe_name = station_name.replace(" ", "_").lower()
    model_path = f"models/model_{safe_name}.pkl" # Pastikan folder 'models' ikut di-upload
    meta_path = f"models/meta_{safe_name}.json"
    
    try:
        model = joblib.load(model_path)
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        return model, metadata
    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.error(f"Error loading model for {station_name}: {e}")
        return None, None

df = load_data()

if df.empty: st.stop()

# --- 3. SIDEBAR (CONTROLS) ---
st.sidebar.title("🎛️ Kontrol Dashboard")

# Filter Stasiun & Tahun (Global)
all_stations = df['station_name'].unique().tolist()
selected_stations = st.sidebar.multiselect("Pilih Stasiun (Filter Global):", all_stations, default=all_stations)

min_year = int(df['year'].min())
max_year = int(df['year'].max())
selected_years = st.sidebar.slider("Rentang Tahun:", min_year, max_year, (min_year, max_year))

# Polutan Fokus
pollutant_options = ['pm25', 'pm10', 'so2', 'co', 'o3', 'no2']
st.sidebar.markdown("---")
main_pollutant = st.sidebar.selectbox("🔎 Polutan Fokus (Utama):", pollutant_options, index=0)

# Export
st.sidebar.markdown("---")
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📥 Download Data CSV", data=csv, file_name='air_quality_data.csv', mime='text/csv')

# --- 4. FILTERING DATA ---
df_filtered = df[
    (df['station_name'].isin(selected_stations)) &
    (df['year'] >= selected_years[0]) & 
    (df['year'] <= selected_years[1])
]

# --- 5. MAIN CONTENT ---
st.title("☁️ Dashboard SkySentinel \nMachine Learning Models \nfor Air Quality Forecasting\n")
st.markdown("Analisis interaktif kualitas udara berdasarkan data historis DKI Jakarta 2021-2025.")

# === KPI CARDS ===
if not df_filtered.empty:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(f"Rata-rata {main_pollutant.upper()}", f"{df_filtered[main_pollutant].mean():.2f}")
    with kpi2:
        st.metric(f"Max {main_pollutant.upper()}", f"{df_filtered[main_pollutant].max():.2f}")
    with kpi3:
        try: dom_cat = df_filtered['category'].mode()[0]
        except: dom_cat = "-"
        st.metric("Kategori Dominan", dom_cat)
    with kpi4:
        worst_st = df_filtered.groupby('station_name')[main_pollutant].mean().idxmax()
        st.metric("Stasiun Terpolusi", worst_st.split('(')[0])
    st.markdown("---")

# === TABS ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🤖 Prediksi (AI)", 
    "📈 Tren", 
    "📊 Kategori & Korelasi", 
    "🏆 Peringkat", 
    "📅 Pola Waktu", 
    "📉 Tahunan",
    "🔥 Risiko"
])

# --- TAB 1: AI PREDICTION (MULTI-STATION) ---
with tab1:
    st.subheader("🤖 Forecasting Cerdas (Multi-Station)")
    
    # 1. User Memilih Stasiun untuk Diprediksi
    sim_stations_list = df['station_name'].unique().tolist()
    target_station = st.selectbox("Pilih Stasiun untuk Prediksi:", sim_stations_list)
    
    # 2. Load Model KHUSUS untuk stasiun tersebut
    model, metadata = load_model_for_station(target_station)
    
    col_pred_1, col_pred_2 = st.columns([1, 2])
    
    with col_pred_1:
        if model is not None:
            # Tampilkan Info Model yang Sesuai
            st.success(f"✅ Model Khusus **{target_station}** Ditemukan!")
            st.info(f"**Performa Model:**\n"
                    f"🔹 MAE: {metadata['mae']:.2f}\n"
                    f"🔹 RMSE: {metadata['rmse']:.2f}\n"
                    f"🔹 MAPE: {metadata['mape']:.2f}\n"
                    f"🔹 Fitur: {metadata['n_lags']} Hari Terakhir")
            
            forecast_days = st.slider("Jumlah Hari Prediksi:", 1, 7, 3)
            predict_btn = st.button("🚀 Jalankan Prediksi")
        else:
            # Fallback jika model belum dilatih untuk stasiun itu
            st.warning(f"⚠️ Model khusus untuk **{target_station}** belum tersedia/dilatih.")
            st.markdown("Silakan jalankan script training di Notebook untuk menghasilkan file `.pkl` stasiun ini.")
            predict_btn = False

    with col_pred_2:
        if predict_btn and model is not None:
            # Ambil data historis stasiun tersebut
            df_target = df[df['station_name'] == target_station].sort_values('Date')
            valid_data = df_target.dropna(subset=['pm25'])
            n_lags = metadata['n_lags']
            
            if len(valid_data) >= n_lags:
                last_window = valid_data['pm25'].values[-n_lags:]
                last_date = valid_data['Date'].max()
                
                # Forecasting Loop
                current_input = last_window.copy()
                future_preds = []
                future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
                
                for _ in range(forecast_days):
                    pred_input = current_input.reshape(1, -1)
                    # Pastikan nama kolom fitur sesuai metadata agar tidak warning
                    input_df = pd.DataFrame(pred_input, columns=metadata['feature_names'])
                    
                    next_val = model.predict(input_df)[0]
                    future_preds.append(next_val)
                    current_input = np.append(current_input[1:], next_val)
                
                # Visualisasi (Sama seperti sebelumnya)
                df_forecast = pd.DataFrame({'Tanggal': future_dates, 'PM2.5': future_preds, 'Tipe': 'PREDIKSI (AI)'})
                df_hist = valid_data.tail(30)[['Date', 'pm25']].rename(columns={'Date': 'Tanggal', 'pm25': 'PM2.5'})
                df_hist['Tipe'] = 'HISTORIS'
                
                df_comb = pd.concat([df_hist, df_forecast])
                
                fig = px.line(df_comb, x='Tanggal', y='PM2.5', color='Tipe', markers=True, 
                              color_discrete_map={'HISTORIS': 'gray', 'PREDIKSI (AI)': '#00CC96'},
                              title=f"Prediksi Kualitas Udara: {target_station}")
                
                # Fix error timestamp pandas 2.0
                last_date_num = pd.to_datetime(last_date).timestamp() * 1000
                fig.add_vline(x=last_date_num, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_forecast.style.format({'PM2.5': '{:.2f}'}))
            else:
                st.error("Data historis tidak cukup untuk prediksi.")

# --- TAB 2-7: EXISTING ANALYTICS ---

with tab2: # Tren
    st.subheader("📈 Tren Dinamika Harian")
    sel_pollutants = st.multiselect("Pilih Parameter:", pollutant_options, default=pollutant_options)
    if sel_pollutants:
        df_d = df_filtered.groupby('Date')[sel_pollutants].mean().reset_index()
        df_melt = df_d.melt(id_vars='Date', var_name='Polutan', value_name='Konsentrasi')
        fig_trend = px.line(df_melt, x='Date', y='Konsentrasi', color='Polutan', template='plotly_white')
        fig_trend.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_trend, use_container_width=True)

with tab3: # Kategori & Korelasi
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🍩 Proporsi Kategori")
        cat_counts = df_filtered['category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig_pie = px.pie(cat_counts, values='Count', names='Category', hole=0.4, color='Category',
                         color_discrete_map={'BAIK': 'green', 'SEDANG': 'blue', 'TIDAK SEHAT': 'orange', 'SANGAT TIDAK SEHAT': 'red', 'BERBAHAYA': 'black'})
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        st.subheader("🔗 Korelasi Polutan")
        corr = df_filtered[pollutant_options].corr()
        fig_corr = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    st.divider()
    st.subheader(f"📦 Distribusi Data {main_pollutant.upper()} (Boxplot)")
    fig_box = px.box(df_filtered, x='category', y=main_pollutant, color='category', points="outliers")
    st.plotly_chart(fig_box, use_container_width=True)

with tab4: # Peringkat
    st.subheader(f"🏆 Peringkat Stasiun ({main_pollutant.upper()})")
    if not df_filtered.empty:
        df_rank = df_filtered.groupby('station_name')[main_pollutant].mean().reset_index().sort_values(main_pollutant)
        fig_rank = px.bar(df_rank, x=main_pollutant, y='station_name', orientation='h', color=main_pollutant, color_continuous_scale='RdYlGn_r', text_auto='.1f')
        st.plotly_chart(fig_rank, use_container_width=True)

with tab5: # Pola Waktu
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📅 Pola Bulanan**")
        df_m = df_filtered.groupby('month_name')[main_pollutant].mean().reset_index()
        fig_m = px.area(df_m, x='month_name', y=main_pollutant, markers=True)
        st.plotly_chart(fig_m, use_container_width=True)
    with c2:
        st.markdown("**📆 Pola Mingguan**")
        df_w = df_filtered.groupby('day_name')[main_pollutant].mean().reset_index()
        fig_w = px.line(df_w, x='day_name', y=main_pollutant, markers=True)
        fig_w.update_layout(yaxis_range=[0, df_w[main_pollutant].max()*1.2])
        st.plotly_chart(fig_w, use_container_width=True)

with tab6: # Tahunan
    st.subheader("📊 Komparasi Tahunan")
    df_y = df_filtered.groupby('year')[pollutant_options].mean().reset_index()
    df_melt_y = df_y.melt(id_vars='year', var_name='Polutan', value_name='Konsentrasi')
    fig_bar_y = px.bar(df_melt_y, x='year', y='Konsentrasi', color='Polutan', barmode='group', text_auto='.1f')
    fig_bar_y.update_xaxes(type='category')
    st.plotly_chart(fig_bar_y, use_container_width=True)

with tab7: # Risiko
    st.subheader(f"🔥 Heatmap Risiko ({main_pollutant.upper()})")
    hm_data = df_filtered.pivot_table(index='year', columns='month_name', values=main_pollutant, aggfunc='mean')
    fig_heat = px.imshow(hm_data, color_continuous_scale='RdBu_r', text_auto='.0f', aspect="auto")
    st.plotly_chart(fig_heat, use_container_width=True)