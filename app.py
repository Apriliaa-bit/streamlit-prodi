import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Rekomendasi Program Studi", layout="centered")

# Sidebar navigasi
with st.sidebar:
    selected = option_menu(
        'Rekomendasi Program Studi UNMA',
        ['Bakat & Minat', 'Prospek Kerja'],
        default_index=0
    )

# Load model & encoder untuk bakat minat
with open("model_bakat_minat.pkl", "rb") as f:
    model_bakat_minat = pickle.load(f)
with open("encoder_bakat.pkl", "rb") as f:
    le_bakat = pickle.load(f)
with open("encoder_minat.pkl", "rb") as f:
    le_minat = pickle.load(f)
with open("encoder_prodi.pkl", "rb") as f:
    le_prodi = pickle.load(f)

# Load model & encoder untuk prospek kerja
with open("model_prospek_pekerjaan.pkl", "rb") as f:
    model_prospek = pickle.load(f)
with open("encoder_bidang.pkl", "rb") as f:
    le_bidang = pickle.load(f)
with open("encoder_pekerjaan.pkl", "rb") as f:
    le_pekerjaan = pickle.load(f)

# Dataset skala untuk tampilkan info tambahan (jika ada)
df_skor = pd.read_excel("Bakat Minat.xlsx")
df_prospek = pd.read_excel("Prospek Kerja.xlsx")  # Sudah disesuaikan strukturnya

deskripsi_skala = {
    1: "Sangat Cocok",
    2: "Cukup Cocok",
    3: "Kurang Cocok",
    4: "Tidak Cocok"
}

# Tampilan umum
st.markdown("""
    <style>
    .stSelectbox div[data-baseweb="select"] {
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Judul halaman berubah tergantung opsi
if selected == 'Bakat & Minat':
    st.title("Rekomendasi Program Studi Berdasarkan Bakat & Minat")
elif selected == 'Prospek Kerja':
    st.title("Rekomendasi Program Studi Berdasarkan Prospek Kerja")

st.markdown("---")


# Opsi: Berdasarkan Bakat & Minat
if selected == 'Bakat & Minat':
    bakat_options = le_bakat.classes_
    minat_options = le_minat.classes_

    bakat = st.selectbox("📌Pilih Bakat", bakat_options)
    minat = st.selectbox("🎯Pilih Minat", minat_options)

    if st.button("🚀Rekomendasikan Program Studi", key="btn_bakat_minat"):
        encoded_bakat = le_bakat.transform([bakat])[0]
        encoded_minat = le_minat.transform([minat])[0]

        pred = model_bakat_minat.predict(np.array([[encoded_bakat, encoded_minat]]))
        prodi = le_prodi.inverse_transform(pred)[0]

        hasil = df_skor[
            (df_skor['Bakat'].str.lower() == bakat.lower()) &
            (df_skor['Minat'].str.lower() == minat.lower()) &
            (df_skor['Program Studi'].str.lower() == prodi.lower())
        ]

        if not hasil.empty:
            skala = int(hasil.iloc[0]['Skala Kecocokan'])
            skala_desc = deskripsi_skala.get(skala, "tidak diketahui")

            if skala in [1, 2]:
                st.success(f"""
                **Rekomendasi Program Studi berdasarkan Bakat dan Minat Anda adalah :**  
                📚 **{prodi}**  
                ✅ **Skala Kecocokan : {skala} ({skala_desc})**
                """)
            else:
                st.warning(f"""
                ⚠️ Kombinasi Bakat dan Minat Anda tidak ideal untuk Program Studi manapun.  
                Namun berdasarkan Minat yang Anda pilih, Program Studi yang masih relevan adalah :  
                📚 **{prodi}**  
                ❌ **Skala Kecocokan : {skala} ({skala_desc})**
                """)
        else:
            st.error("⚠️ Kombinasi bakat dan minat Anda belum terdaftar di dataset.")


# Opsi: Berdasarkan Prospek Kerja
elif selected == 'Prospek Kerja':
    
    # Dropdown dinamis berdasarkan bidang
    bidang_options = df_prospek['Bidang'].unique()
    bidang = st.selectbox("📌Pilih Bidang Kerja", bidang_options)

    # Filter pekerjaan sesuai bidang yang dipilih
    pekerjaan_filtered = df_prospek[df_prospek['Bidang'] == bidang]['Pekerjaan'].unique()
    pekerjaan = st.selectbox("🎯Pilih Pekerjaan Impian", pekerjaan_filtered)

    if st.button("🚀Rekomendasikan Program Studi", key="btn_prospek"):
        try:
            # Encoding input
            encoded_bidang = le_bidang.transform([bidang])[0]
            encoded_pekerjaan = le_pekerjaan.transform([pekerjaan])[0]

            # Prediksi prodi
            pred_prodi = model_prospek.predict(np.array([[encoded_bidang, encoded_pekerjaan]]))
            prodi_hasil = le_prodi.inverse_transform(pred_prodi)[0]

            # Cek apakah hasil cocok dengan data asli
            hasil_prospek = df_prospek[
                (df_prospek['Bidang'] == bidang) &
                (df_prospek['Pekerjaan'] == pekerjaan) &
                (df_prospek['Program Studi'] == prodi_hasil)
            ]

            if not hasil_prospek.empty:
                st.success(f"""
                **Rekomendasi Program Studi berdasarkan Pekerjaan Impian Anda adalah :**  
                📚 **{prodi_hasil}**
                """)
            else:
                st.warning(f"""
                ⚠️ Kombinasi bidang dan pekerjaan Anda belum tersedia di dataset.  
                Tapi model merekomendasikan: 📚 **{prodi_hasil}**
                """)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat proses rekomendasi: {e}")
