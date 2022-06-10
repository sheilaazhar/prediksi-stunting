#Library
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model

from numpy.random import seed
seed(12345)
import tensorflow as tf  
tf.random.set_seed(12345) 
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from scikeras.wrappers import KerasClassifier
from imblearn.pipeline import Pipeline as imbpipeline
from pickle import load, dump

st.set_page_config(page_title="Prediksi Stunting",layout='wide', page_icon=':rocket:')
st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',unsafe_allow_html=True)
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown(f"""
    <style>
        .appview-container .main .block-container{{
            padding-top: {3}rem;
        }}
    </style>""",
    unsafe_allow_html=True,
)

# st.title("Form Prediksi Stunting")
menu = ["Home","Pelatihan Model","Prediksi"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.markdown('''
    <h1 style="font-size:30px; text-align: center; margin-bottom:5px">PREDIKSI KONDISI STUNTING PADA BALITA 
    MENGGUNAKAN METODE LONG SHORT TERM MEMORY (LSTM)</h1>
    ''',unsafe_allow_html=True)
    img1, img2, img3 = st.columns([1,4,1])
    with img1:
        st.write('')
    with img2:
        image = Image.open('stunting.jpg')
        st.image(image, caption='Ilustrasi Stunting')
    with img3:
        st.write(' ')
    st.markdown('''
    <p align = "justify">Stunting telah lama menjadi isu prioritas nasional akibat tingginya prevalensi stunting di Indonesia. Berdasarkan data Riskesdas, angka stunting di Indonesia memang telah mengalami penurunan dari tahun ke tahun, yaitu dari 37,2 persen (2013), 30,8 persen (2018), 27,7 persen (2019), dan menjadi 24,4 persen (2021). Namun angka ini masih jauh dari total presentase standar World Health Organization (WHO), yaitu maksimal 20 persen. Sehingga pemerintah memiliki target menurunkan angka stunting menjadi 14% di tahun 2024.
Penelitian ini bertujuan untuk membuat model yang dapat memprediksi stunting pada balita yang diharapkan dapat membantu untuk mencegah terjadinya stunting. Dalam memprediksi stunting, digunakan sebuah algoritma deep learning, yaitu Long Short Term Memory (LSTM).
Data yang digunakan pada penelitian ini diambil dari data Indonesia Family Life Survey (IFLS) gelombang 1, 3, 4, dan 5. Untuk menghasilkan model yang optimal, dilakukan pengujian hyperparameter pada jumlah lapisan LSTM sebanyak 1, 2, dan 3 lapis. Lalu jumlah unit LSTM sebanyak 16, 32, dan 64. Pengujian tersebut menghasilkan model paling optimal dengan jumlah lapisan LSTM sebanyak 1 lapis, jumlah unit LSTM sebanyak 32, optimizer RMSprop, activation sigmoid, learning rate 0.001, dropout 0.5, batch size 32, dan max epoch 100. Model LSTM dievaluasi menggunakan 10-fold cross validation dan menghasilkan rata-rata f1-score 76,22%.
</p>
    ''',unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="row">
            <div class="col-sm-4">
                <div class="card text-center bg-light mb-3 mt-5" style="width: 18rem;">
                <div class="card-header">
                    PENULIS
                </div>
                    <div class="card-body">
                        <p class="card-text">Sheila Azhar Almufarida <br>NPM : 140810180001</p>
                    </div>
                </div>
            </div>
            <div class="col-sm-4">
                <div class="card text-center bg-light mb-3 mt-5" style="width: 18rem;">
                <div class="card-header">
                    PEMBIMBING 1
                </div>
                    <div class="card-body">
                        <p class="card-text">Dr. Intan Nurma Yulita, MT <br>NIP : 19850704 201504 2 003</p>
                    </div>
                </div>
            </div>
            <div class="col-sm-4">
                <div class="card text-center bg-light mb-3 mt-5" style="width: 18rem;">
                <div class="card-header">
                    PEMBIMBING 2
                </div>
                    <div class="card-body">
                        <p class="card-text">Aditya Pradana, S. T., M. Eng. <br>NIP : 19841211 201504 1 002</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True
    )

elif choice == "Pelatihan Model":
    st.markdown(
        """
        <center><h1 class="title">Pelatihan Model LSTM</h1><center/>
        """, unsafe_allow_html=True)

    st.markdown(""" #### Dataset """)
    dataset = pd.read_csv('data/dataset_stunting_ifls.csv')
    st.write(dataset)

    st.sidebar.subheader('Hyperparameter')
    lapisan = st.sidebar.selectbox("Jumlah Lapisan LSTM", ('1','2','3'))
    unit = st.sidebar.selectbox("Jumlah Unit LSTM", ('16','32','64'))
    if st.sidebar.button("Train"):
        unit = int(unit)

        #Preprocessing
        #Menyamakan satuan data
        dataset.loc[dataset['satuan_usia_kehamilan'] == 5.0, 'usia_kehamilan_saat_persalinan'] = (dataset['usia_kehamilan_saat_persalinan'] * 4) #mengubah data usia bulan ke minggu
        dataset.loc[dataset['satuan_usia_minum_air'] == 5.0, 'usia_minum_air_pertamakali'] = (dataset['usia_minum_air_pertamakali'] * 30) #mengubah usia dari bulan ke hari
        dataset.loc[dataset['satuan_usia_minum_air'] == 6.0, 'usia_minum_air_pertamakali'] = (dataset['usia_minum_air_pertamakali'] * 30) #mengubah usia dari bulan ke hari
        dataset.loc[dataset['satuan_usia_minum_air'] == 4.0, 'usia_minum_air_pertamakali'] = (dataset['usia_minum_air_pertamakali'] * 7) #mengubah usia dari minggu ke hari
        dataset.loc[dataset['satuan_usia_minum_air'] == 2.0, 'usia_minum_air_pertamakali'] = (dataset['usia_minum_air_pertamakali'] * 7) #mengubah usia dari minggu ke hari
        dataset.loc[dataset['satuan_usia_makan_pertama'] == 4.0, 'usia_makan_pertama_selain_asi'] = (dataset['usia_makan_pertama_selain_asi'] * 7) #mengubah usia dari minggu ke hari
        dataset.loc[dataset['satuan_usia_makan_pertama'] == 5.0, 'usia_makan_pertama_selain_asi'] = (dataset['usia_makan_pertama_selain_asi'] * 30) #mengubah usia dari bulan ke hari
        #Menghapus data tidak logis
        dataset.loc[dataset['tinggi_badan'] == 0, 'tinggi_badan'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['tinggi_badan'] > 100, 'tinggi_badan'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['tinggi_badan'] < 20, 'tinggi_badan'] = np.NaN #mengubah dataset tidak logis
        dataset.loc[dataset['berat_badan'] == 0, 'berat_badan'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['berat_badan'] > 100, 'berat_badan'] = np.NaN
        dataset.loc[dataset['tinggi_badan_ayah'] > 200, 'tinggi_badan_ayah'] = np.NaN #mengubah data yang tidak logis
        dataset.loc[dataset['tinggi_badan_ayah'] == 0, 'tinggi_badan_ayah'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['tinggi_badan_ayah'] < 100, 'tinggi_badan_ayah'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['tinggi_badan_ibu'] == 0, 'tinggi_badan_ibu'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['tinggi_badan_ibu'] > 200, 'tinggi_badan_ibu'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['tinggi_badan_ibu'] < 100, 'tinggi_badan_ibu'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['usia_ibu_melahirkan'] < 10, 'usia_ibu_melahirkan'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['usia_kehamilan_saat_persalinan'] > 50, 'usia_kehamilan_saat_persalinan'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['usia_kehamilan_saat_persalinan'] < 20, 'usia_kehamilan_saat_persalinan'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['berat_badan_bayi'] == 0, 'berat_badan_bayi'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['berat_badan_bayi'] > 9, 'berat_badan_bayi'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['usia_minum_air_pertamakali'] == 0, 'usia_minum_air_pertamakali'] = np.NaN #mengubah data tidak logis
        dataset.loc[dataset['usia_makan_pertama_selain_asi'] == 0, 'usia_makan_pertama_selain_asi'] = np.NaN #mengubah data tidak logis
        dataset['usia_kehamilan_saat_persalinan'] = dataset['usia_kehamilan_saat_persalinan'].round()
        dataset['usia_minum_air_pertamakali'] = dataset['usia_minum_air_pertamakali'].round()
        dataset['usia_makan_pertama_selain_asi'] = dataset['usia_makan_pertama_selain_asi'].round()
        #mengatasi data yang tidak diketahui
        dataset.loc[dataset['pendidikan_ayah'] == 7, 'pendidikan_ayah'] = np.NaN #mengubah data yang tidak menjawab
        dataset.loc[dataset['pendidikan_ayah'] == 9, 'pendidikan_ayah'] = np.NaN #mengubah data yang missing
        dataset.loc[dataset['pendidikan_ayah'] >= 98, 'pendidikan_ayah'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['apakah_anak_imunisasi_bcg'] == 8, 'apakah_anak_imunisasi_bcg'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['apakah_anak_Imunisasi_polio'] == 8, 'apakah_anak_Imunisasi_polio'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['apakah_anak_imunisasi_dpt'] == 8, 'apakah_anak_imunisasi_dpt'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['apakah_anak_imunisasi_campak'] == 8, 'apakah_anak_imunisasi_campak'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['apakah_anak_imunisasi_hb'] == 8, 'apakah_anak_imunisasi_hb'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['apakah_kebiasaan_merokok_sampai_sekarang'] > 3, 'apakah_kebiasaan_merokok_sampai_sekarang'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['sumber_air_minum_utama'] == 99, 'sumber_air_minum_utama'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['tempat_buang_air_besar'] == 99, 'tempat_buang_air_besar'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['tempat_pembuangan_limbah'] == 99, 'tempat_pembuangan_limbah'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['tempat_pembuangan_sampah'] == 99, 'tempat_pembuangan_sampah'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['pendidikan_ibu'] == 7, 'pendidikan_ibu'] = np.NaN #mengubah data yang tidak menjawab
        dataset.loc[dataset['pendidikan_ibu'] == 9, 'pendidikan_ibu'] = np.NaN #mengubah data yang missing
        dataset.loc[dataset['pendidikan_ibu'] >= 98, 'pendidikan_ibu'] = np.NaN #mengubah data yang tidak tahu
        dataset.loc[dataset['persepsi_ibu_bayi_lebih_besar'] >= 8, 'persepsi_ibu_bayi_lebih_besar'] = np.NaN #mengubah data yang tidak tahu
        dataset.loc[dataset['tempat_pemeriksaan_kehamilan'] == '99', 'tempat_pemeriksaan_kehamilan'] = np.nan #mengubah data yang tidak tahu
        dataset.loc[dataset['tempat_pemeriksaan_kehamilan'] == 'Z', 'tempat_pemeriksaan_kehamilan'] = np.NaN #mengubah data yang missing
        dataset.loc[dataset['apakah_pernah_menyusui'] >= 8, 'apakah_pernah_menyusui'] = np.NaN #mengubah data yang tidak diketahui & missing
        dataset.loc[dataset['anc_berat'] >= 8, 'anc_berat'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['anc_tinggi'] >= 8, 'anc_tinggi'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['anc_td'] >= 8, 'anc_td'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['anc_teshb'] >= 8, 'anc_teshb'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['anc_tfu'] >= 8, 'anc_tfu'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['anc_djj'] >= 8, 'anc_djj'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['anc_pd'] >= 8, 'anc_pd'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['anc_panggulluar'] >= 8, 'anc_panggulluar'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['imunisasi_tt'] >= 8, 'imunisasi_tt'] = np.NaN #mengubah data yang tidak diketahui
        dataset.loc[dataset['satuan_usia_minum_air'] >= 9, 'satuan_usia_minum_air'] = np.nan #mengubah satuan yang tidak diketahui (tidak tahu usianya) menjadi NaN
        dataset.loc[dataset['usia_makan_pertama_selain_asi'] >= 70, 'usia_makan_pertama_selain_asi'] = np.nan #mengubah usia yang tidak diketahui menjadi NaN
        dataset.loc[dataset['usia_minum_air_pertamakali'] >= 70, 'usia_minum_air_pertamakali'] = np.nan #mengubah usia yang tidak diketahui menjadi NaN
        #memperbaiki inkonsistensi data
        dataset.loc[dataset['tempat_pemeriksaan_kehamilan'] == '1', 'tempat_pemeriksaan_kehamilan'] = 'A' #menyamakan kategori data
        dataset.loc[dataset['tempat_pemeriksaan_kehamilan'] == '2', 'tempat_pemeriksaan_kehamilan'] = 'B' #menyamakan kategori data
        dataset.loc[dataset['tempat_pemeriksaan_kehamilan'] == '3', 'tempat_pemeriksaan_kehamilan'] = 'C' #menyamakan kategori data
        dataset.loc[dataset['tempat_pemeriksaan_kehamilan'] == '4', 'tempat_pemeriksaan_kehamilan'] = 'E' #menyamakan kategori data
        dataset.loc[dataset['tempat_pemeriksaan_kehamilan'] == '5', 'tempat_pemeriksaan_kehamilan'] = 'F' #menyamakan kategori data
        dataset.loc[dataset['tempat_pemeriksaan_kehamilan'] == '6', 'tempat_pemeriksaan_kehamilan'] = 'G' #menyamakan kategori data
        dataset.loc[dataset['tempat_pemeriksaan_kehamilan'] == '7', 'tempat_pemeriksaan_kehamilan'] = 'V' #menyamakan kategori data other
        dataset.loc[dataset['tempat_pemeriksaan_kehamilan'] == '8', 'tempat_pemeriksaan_kehamilan'] = 'V' #menyamakan kategori data other
        dataset.loc[dataset['tempat_pemeriksaan_kehamilan'] == 'H', 'tempat_pemeriksaan_kehamilan'] = 'V' #menyamakan kategori data other
        
        #Handle missing value
        # Hapus kolom yang memiliki nilai null 
        threshold=0.8
        dataset = dataset[dataset.columns[dataset.isnull().mean() < threshold]]
        # Mengisi missing value pada data normal distribution dengan mean
        num1 = ['tinggi_badan','berat_badan', 'tinggi_badan_ayah','tinggi_badan_ibu','berat_badan_bayi']
        impute_mean = SimpleImputer(strategy='mean', missing_values=np.nan)
        impute_mean = impute_mean.fit(dataset[num1])
        dataset[num1] = impute_mean.transform(dataset[num1])
        # Mengisi missing value pada data skew distribution dengan median
        num2 = ['jumlah_anggota_kel','usia_ibu_melahirkan','usia_makan_pertama_selain_asi', 'frekuensi_anc_tm1','frekuensi_anc_tm2','frekuensi_anc_tm3','jumlah_fe_diminum_selama_hamil','usia_kehamilan_saat_persalinan','usia_minum_air_pertamakali']
        impute_median = SimpleImputer(strategy='median', missing_values=np.nan)
        impute_median = impute_median.fit(dataset[num2])
        dataset[num2] = impute_median.transform(dataset[num2])
        # Mengisi missing value pada data kategorik dengan modus
        category = ['jenis_kelamin','apakah_anak_minum_vitA_6bln_terakhir','apakah_anak_imunisasi_bcg','apakah_anak_Imunisasi_polio','apakah_anak_imunisasi_dpt','apakah_anak_imunisasi_campak','apakah_anak_imunisasi_hb',
                                            'pendidikan_ayah','apakah_bekerja_ayah','apakah_kebiasaan_merokok_sampai_sekarang','sumber_air_minum_utama', 'tempat_buang_air_besar', 'tempat_pembuangan_limbah', 'tempat_pembuangan_sampah', 
                                            'apakah_bekerja_ibu','pendidikan_ibu','pengobatan_anemia','pengobatan_hipertensi','pengobatan_dm','apakah_lahir_kembar','apakah_pernah_menyusui',
                                            'persepsi_ibu_bayi_lebih_besar','tempat_pemeriksaan_kehamilan','apakah_ada_komplikasi_kehamilan','anc_berat','anc_tinggi','anc_td','anc_teshb','anc_tfu','anc_djj','anc_pd','anc_panggulluar','imunisasi_tt']
        impute_mode = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        impute_mode = impute_mode.fit(dataset[category])
        dataset[category] = impute_mode.transform(dataset[category])

        #Transformasi data (one hot encoding)
        cat_columns = ['jenis_kelamin','apakah_anak_minum_vitA_6bln_terakhir','apakah_anak_imunisasi_bcg','apakah_anak_Imunisasi_polio','apakah_anak_imunisasi_dpt','apakah_anak_imunisasi_campak','apakah_anak_imunisasi_hb','apakah_bekerja_ayah','apakah_kebiasaan_merokok_sampai_sekarang','apakah_bekerja_ibu','pengobatan_anemia','pengobatan_hipertensi','pengobatan_dm','apakah_lahir_kembar','apakah_pernah_menyusui','anc_berat','anc_tinggi','anc_td','anc_teshb','anc_tfu','anc_djj','anc_pd','anc_panggulluar','imunisasi_tt','pendidikan_ayah','sumber_air_minum_utama', 'tempat_buang_air_besar', 'tempat_pembuangan_limbah', 'tempat_pembuangan_sampah', 'pendidikan_ibu','persepsi_ibu_bayi_lebih_besar']
        all_cat_columns = ['jenis_kelamin','apakah_anak_minum_vitA_6bln_terakhir', 'apakah_anak_imunisasi_bcg', 'apakah_anak_Imunisasi_polio', 'apakah_anak_imunisasi_dpt', 'apakah_anak_imunisasi_campak', 'apakah_anak_imunisasi_hb', 'apakah_bekerja_ayah', 'apakah_kebiasaan_merokok_sampai_sekarang', 'apakah_bekerja_ibu', 'pengobatan_anemia', 'pengobatan_hipertensi', 'pengobatan_dm', 'apakah_lahir_kembar', 'apakah_pernah_menyusui', 'anc_berat', 'anc_tinggi', 'anc_td', 'anc_teshb', 'anc_tfu', 'anc_djj', 'anc_pd', 'anc_panggulluar', 'imunisasi_tt', 'pendidikan_ayah','sumber_air_minum_utama', 'tempat_buang_air_besar', 'tempat_pembuangan_limbah',  'tempat_pembuangan_sampah', 'pendidikan_ibu','persepsi_ibu_bayi_lebih_besar','tempat_pemeriksaan_kehamilan','apakah_ada_komplikasi_kehamilan']
        X_dataset = dataset.iloc[:, :-1]
        y_dataset = dataset.iloc[:, 51]
        X_dataset = pd.get_dummies(X_dataset, prefix_sep="__", columns=cat_columns)
        X_dataset['tempat_pemeriksaan_kehamilan'] = X_dataset['tempat_pemeriksaan_kehamilan'].astype(str)
        X_dataset['tempat_pemeriksaan_kehamilan'] = X_dataset['tempat_pemeriksaan_kehamilan'].apply(list)
        X_dataset['apakah_ada_komplikasi_kehamilan'] = X_dataset['apakah_ada_komplikasi_kehamilan'].apply(list)
        OneHt_periksa = X_dataset['tempat_pemeriksaan_kehamilan'].str.join('|').str.get_dummies()
        OneHt_periksa = OneHt_periksa.add_prefix('tempat_pemeriksaan_kehamilan__')
        X_dataset = pd.concat([X_dataset,OneHt_periksa], axis=1)
        OneHt_komplikasi = X_dataset['apakah_ada_komplikasi_kehamilan'].str.join('|').str.get_dummies()
        OneHt_komplikasi = OneHt_komplikasi.add_prefix('apakah_ada_komplikasi_kehamilan__')
        X_dataset = pd.concat([X_dataset,OneHt_komplikasi], axis=1)
        X_dataset = X_dataset.drop(columns=['tempat_pemeriksaan_kehamilan','apakah_ada_komplikasi_kehamilan','satuan_usia_kehamilan','satuan_usia_minum_air'])
        cat_dummies = [col for col in X_dataset if "__" in col and col.split("__")[0] in all_cat_columns]
        processed_columns = list(X_dataset.columns[:])
        #save info data
        dump(cat_dummies, open('list_dummies.pkl', 'wb'))
        dump(processed_columns, open('columns.pkl', 'wb'))
        X_dataset = X_dataset.drop(columns=['pidlink'])
        dataset = pd.concat([X_dataset, y_dataset], axis = 1)
        dump(impute_mean, open('mean.pkl', 'wb'))
        dump(impute_median, open('median.pkl', 'wb'))
        dump(impute_mode, open('mode.pkl', 'wb'))
        scaler = MinMaxScaler()
        scaler.fit(X_dataset)
        dump(scaler, open('scaler.pkl', 'wb'))

        #memisahkan data prediktor dan target
        X = dataset.drop(['label'], axis=1)
        y = dataset.label

        #Handling imbalanced class
        sm = SMOTE(random_state=5)
        X, y = sm.fit_resample(X, y)

        #cross validation
        def cross_validate(model, _X, _y, cv):
            X = _X
            y = _y
            train_acc = []
            train_prec = []
            train_rec = []
            train_fs = []
            test_acc = []
            test_prec = []
            test_rec = []
            test_fs = []
            conf_matrix = []
            for train_ind, val_ind in cv.split(X, y):
                X_t, y_t = X.iloc[train_ind], y[train_ind]
                pipeline.fit(X_t, y_t)
                y_pred_t = pipeline.predict(X_t)
                train_acc.append(accuracy_score(y_t, y_pred_t))
                train_prec.append(precision_score(y_t, y_pred_t))
                train_rec.append(recall_score(y_t, y_pred_t))
                train_fs.append(f1_score(y_t, y_pred_t))
                X_val, y_val = X.iloc[val_ind], y[val_ind]
                y_pred_val = pipeline.predict(X_val)
                test_acc.append(accuracy_score(y_val, y_pred_val))
                test_prec.append(precision_score(y_val, y_pred_val))
                test_rec.append(recall_score(y_val, y_pred_val))
                test_fs.append(f1_score(y_val, y_pred_val))
                matrix = confusion_matrix(y_val,y_pred_val)
                conf_matrix.append(matrix)
            st.markdown(""" #### Hasil Rata-rata Pengujian 10-Fold Cross Validation """)
            acc_avg = format((np.mean(test_acc))*100, ".2f")
            prec_avg = format((np.mean(test_prec))*100, ".2f")
            rec_avg = format((np.mean(test_rec))*100, ".2f")
            fs_avg = format((np.mean(test_fs))*100, ".2f")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Akurasi", acc_avg + "%")
            a2.metric("Precision", prec_avg + "%")
            a3.metric("Recall", rec_avg + "%")
            a4.metric("F1-score", fs_avg + "%")
            # st.write('Test Accuracy : {}'.format(np.mean(test_acc)))
            # st.write('Test Precision : {}'.format(np.mean(test_prec)))
            # st.write('Test Recall: {}'.format(np.mean(test_rec)))
            # st.write('Test F1-score: {}'.format(np.mean(test_fs)))
            # st.write('Test accuracy of each fold - {}'.format(test_acc))
            # st.write('Test precision of each fold - {}'.format(test_prec))
            # st.write('Test recall of each fold - {}'.format(test_rec))
            # st.write('Test f1-score of each fold - {}'.format(test_fs))
            # st.write('\n')
            cf = plt.figure(figsize = (2, 2))
            sns.heatmap((np.mean(conf_matrix, axis=0)),  cmap= 'PuBu', annot=True, fmt='g', annot_kws={'size':10})
            plt.xlabel('Prediksi', fontsize=8)
            plt.ylabel('Aktual', fontsize=8)
            plt.title("Confusion Matrix", fontsize=8)
            # plt.show();
            p1, p2, p3 = st.columns(3)
            p2.pyplot(cf)
            # st.write('Confusion Matrix of each fold: {}', conf_matrix)
            # for i in range(len(conf_matrix)):
            #     sns.heatmap(conf_matrix[i],  cmap= 'PuBu', annot=True, fmt='g', annot_kws={'size':10})
            #     plt.xlabel('predicted', fontsize=8)
            #     plt.ylabel('actual', fontsize=8)
            #     plt.title("Confusion Matrix", fontsize=8)
            #     plt.show();

        #reshape to 3D
        def reshape_data(X, y=None):
            X = np.reshape(X, (X.shape[0],1, X.shape[1]))
            return X
        reshape = FunctionTransformer(func=reshape_data)

        #lstm
        def create_model():
            model = Sequential()

            if lapisan == '1' :
                model.add(LSTM(unit, input_shape=(1,155)))
                model.add(Dropout(0.5))
            
            elif lapisan == '2' :
                model.add(LSTM(unit, input_shape=(1,155), return_sequences = True))
                model.add(Dropout(0.5))
                model.add(LSTM(unit))
                model.add(Dropout(0.5))
            
            else :
                model.add(LSTM(unit, input_shape=(1,155), return_sequences = True))
                model.add(Dropout(0.5))
                model.add(LSTM(unit, return_sequences = True))
                model.add(Dropout(0.5))
                model.add(LSTM(unit))
                model.add(Dropout(0.5))

            #Add output layer
            model.add(Dense(1, activation='sigmoid'))

            model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])

            return model

        #kfold
        estimators = []
        estimators.append(('normalize', MinMaxScaler(feature_range=(0, 1))))
        estimators.append(('reshape', reshape))
        estimators.append(('lstm', KerasClassifier(model=create_model, epochs=100, batch_size=32, callbacks={"cp": ModelCheckpoint, "es": EarlyStopping},
                                                callbacks__cp__filepath='model/model.hdf5',
                                                callbacks__cp__monitor="accuracy",
                                                callbacks__cp__save_best_only=True,
                                                callbacks__es__monitor="loss",
                                                callbacks__es__patience=10,
                                                verbose=0,
                                                shuffle=False)))
        pipeline = imbpipeline(estimators)
        kfold = KFold(n_splits=10, random_state=20, shuffle=True)
        results = cross_validate(pipeline, X, y, kfold)

        st.write(results)
    
elif choice == "Prediksi":
    rad = st.sidebar.radio("Pilihan", ["Form", "File"])

    if rad =="Form":
        st.markdown(
            """
            <center><h1 class="title">Form Prediksi Stunting</h1><center/>
            """, unsafe_allow_html=True)
        
        dataframe = pd.DataFrame()

        jk_opt = {9999:"", "1": "Laki-Laki", "3": "Perempuan"}
        satuanm_opt = {9999:"", 3: "Hari", 4: "Minggu", 5: "Bulan"}
        iyatidak_opt = {9999:"", 1: "Iya", 3: "Tidak"}
        pend_opt = {9999:"", 1:"Tidak sekolah", 2:"SD", 3:"SMP", 4:"SMP Kejuruan", 5:"SMA", 6:"SMK", 11:"Pendidikan A", 12:"Pendidikan B", 13:"Universitas terbuka", 14:"Pesantren", 15:"Pendidikan C", 17:"Sekolah disabilitas", 60:"Diploma (D1,D2,D3)", 61:"S1", 62:"S2", 63:"S3", 72:"Madrasah Ibtidaiyah", 73:"Madrasah Tsanawiyah", 74:"Madrasah Aliyah", 90:"TK", 95:"Lainnya"}
        satuank_opt = {9999:"", 4: "Minggu", 5: "Bulan"}
        ukuran_opt = {9999:"", 1: "Sangat besar", 2: "Besar", 3:"Normal", 4:"Kecil", 5:"Sangat kecil"}
        tempatanc_opt = {9999:"", "A":"RS.Pemerintah", "B":"RS.Swasta", "C":"Puskesmas", "D":"Polindes", "E":"Klinik Swasta", "F":"PMB", "G":"Rumah dukun bayi", "I":"Posyandu", "J":"Dokter spesialis", "K":"RS.Bersalin", "V":"Lainnya"}
        komplikasi_opt = {9999:"", "A":"Pembengkakan kaki", "B":"Kesulitan melihat siang hari", "C":"Kesulitan melihat malam hari", "D":"Pendarahan", "E":"Demam", "F":"Kejang dan pingsan", "G":"Kesakitan karena mau melahirkan sebelum 9 bulan", "W":"Tidak ada komplikasi"}
        sumberair_opt = {9999:"", 1:"Ledeng", 2:"Sumur/pompa", 3:"Sumur timba/perigi", 4:"Mata air", 5:"Air hujan", 6:"Air sungai", 7:"Kolam/balong/empang", 8:"Bak penampungan", 10:"Air mineral", 95:"Lainnya"}
        bab_opt = {9999:"", 1:"Jamban sendiri dengan septik tank", 2:"Jamban sendiri tanpa septik tank", 3:"Jamban bersama", 4:"Jamban umum", 5:"Kali/sungai/parit", 6:"Kebun/sawah", 7:"Selokan", 9:"Kolam/balong/empang", 10:"Kandang ternak", 11:"Laut/danau", 95:"Lainnya"}
        limbah_opt = {9999:"", 1:"Selokan/got yang mengalir", 2:"Selokan/got tidak mengalir", 3:"Lubang permanen", 4:"Dibuang ke sungai", 5:"Dibuang di samping/belakang rumah/kebun", 7:"Kolam/balong/empang/danau", 9:"Sawah/ladang", 11:"Laut/pantai", 95:"Lainnya"}
        sampah_opt = {9999:"", 1:"Dibuang di tempat/tong sampah dan diangkut petugas", 2:"Dibakar", 3:"Dibuang di sungai/selokan", 4:"Dibuang di pekarangan rumah/kebun, dibiarkan", 5:"Ditimbun di lubang", 7:"Hutan/gunung", 8:"Laut/danau/pantai", 9:"Sawah/ladang", 95:"Lainnya"}

        st.markdown(""" #### DATA ANAK """)
        tinggi_badan = st.number_input("Tinggi badan (cm)", min_value = 0.0, max_value = 100.0, step = 1.0)
        bb1, bb2 = st.columns(2)
        berat_badan = bb1.number_input("Berat badan (kg)", min_value = 0.0, max_value = 20.0, step = 1.0)
        berat_badan_bayi = bb2.number_input("Berat badan saat lahir (kg)", min_value = 0.0, max_value = 10.0, step = 1.0)
        jenis_kelamin = st.selectbox("Jenis kelamin", options=list(jk_opt.keys()), format_func=lambda x: jk_opt.get(x))
        jumlah_anggota_kel = st.number_input("Jumlah Anggota Keluarga", min_value = 0, step = 1)
        makan1, makan2 = st.columns(2)
        usia_makan_pertama_selain_asi = makan1.number_input("Usia makan pertama selain ASI", min_value = 0, step = 1)
        satuan_usia_makan_pertama = makan2.selectbox("Satuan usia makan", options=list(satuanm_opt.keys()), format_func=lambda x: satuanm_opt.get(x))
        minum1, minum2 = st.columns(2)
        usia_minum_air_pertamakali = minum1.number_input("Usia minum air pertama kali", min_value = 0, step = 1)
        satuan_usia_minum_air = minum2.selectbox("Satuan usia minum", options=list(satuanm_opt.keys()), format_func=lambda x: satuanm_opt.get(x))
        apakah_anak_minum_vitA_6bln_terakhir = st.selectbox("Apakah anak minum vitamin A dalam 6 bulan terakhir?", options=list(iyatidak_opt.keys()), format_func=lambda x: iyatidak_opt.get(x))
        st.markdown(""" ###### Imunisasi """)
        # apakah_anak_imunisasi_bcg = st.selectbox("Apakah anak sudah diimunisasi BCG?", ["Iya", "Tidak"], index = 0)
        # apakah_anak_Imunisasi_polio = st.selectbox("Apakah anak sudah diimunisasi Polio?", ["Iya", "Tidak"], index = 0)
        # apakah_anak_imunisasi_dpt = st.selectbox("Apakah anak sudah diimunisasi DPT?", ["Iya", "Tidak"], index = 0)
        # apakah_anak_imunisasi_campak = st.selectbox("Apakah anak sudah diimunisasi Campak?", ["Iya", "Tidak"], index = 0)
        # apakah_anak_imunisasi_hb = st.selectbox("Apakah anak sudah diimunisasi Hepatitis B?", ["Iya", "Tidak"], index = 0)
        bcg, dpt, hb = st.columns(3)
        polio, campak, vaksin = st.columns(3)
        apakah_anak_imunisasi_bcg = bcg.checkbox("BCG")
        apakah_anak_Imunisasi_polio = polio.checkbox("Polio")
        apakah_anak_imunisasi_dpt = dpt.checkbox("DPT")
        apakah_anak_imunisasi_campak = campak.checkbox("Campak")
        apakah_anak_imunisasi_hb = hb.checkbox("Hepatitis B")
        st.markdown(""" --- """)

        st.markdown(""" #### DATA AYAH """)
        tinggi_badan_ayah = st.number_input("Tinggi badan Ayah (cm)", min_value = 0.0, max_value = 200.0, step = 1.0)
        apakah_bekerja_ayah = st.selectbox("Apakah Ayah bekerja?", options=list(iyatidak_opt.keys()), format_func=lambda x: iyatidak_opt.get(x))
        pendidikan_ayah = st.selectbox("Pendidikan terakhir Ayah", options=list(pend_opt.keys()), format_func=lambda x: pend_opt.get(x))
        apakah_kebiasaan_merokok_sampai_sekarang = st.selectbox("Apakah sekarang memiliki kebiasaan merokok?", options=list(iyatidak_opt.keys()), format_func=lambda x: iyatidak_opt.get(x))
        st.markdown(""" --- """)

        st.markdown(""" #### DATA IBU """)
        tinggi_badan_ibu = st.number_input("Tinggi badan Ibu (cm)", min_value = 0.0, max_value = 200.0, step = 1.0)
        apakah_bekerja_ibu = st.selectbox("Apakah Ibu bekerja?", options=list(iyatidak_opt.keys()), format_func=lambda x: iyatidak_opt.get(x))
        pendidikan_ibu = st.selectbox("Pendidikan terakhir Ibu", options=list(pend_opt.keys()), format_func=lambda x: pend_opt.get(x))
        st.markdown(""" ###### Pengobatan Ibu """)
        # pengobatan_anemia = st.selectbox("Pengobatan anemia", ["Iya", "Tidak"], index = 0)
        # pengobatan_hipertensi = st.selectbox("Pengobatan hipertensi", ["Iya", "Tidak"], index = 0)
        # pengobatan_dm = st.selectbox("Pengobatan dm", ["Iya", "Tidak"], index = 0)
        pengobatan_anemia = st.checkbox("Anemia")
        pengobatan_hipertensi = st.checkbox("Hipertensi")
        pengobatan_dm = st.checkbox("Diabetes Melitus")
        usia_ibu_melahirkan = st.number_input("Usia ibu saat persalinan", min_value = 0, max_value = 100, step = 1)
        usiak, usias = st.columns(2)
        usia_kehamilan_saat_persalinan = usiak.number_input("Usia kehamilan ibu saat persalinan", min_value = 0, max_value = 50, step = 1)
        satuan_usia_kehamilan = usias.selectbox("Satuan usia kehamilan", options=list(satuank_opt.keys()), format_func=lambda x: satuank_opt.get(x))
        apakah_lahir_kembar = st.selectbox("Melahirkan tunggal/kembar?", options=list(iyatidak_opt.keys()), format_func=lambda x: iyatidak_opt.get(x))
        persepsi_ibu_bayi_lebih_besar = st.selectbox("Persepsi ibu ukuran bayi", options=list(ukuran_opt.keys()), format_func=lambda x: ukuran_opt.get(x))
        apakah_pernah_menyusui = st.selectbox("Apakah pernah menyusui?", options=list(iyatidak_opt.keys()), format_func=lambda x: iyatidak_opt.get(x))
        jumlah_fe_diminum_selama_hamil = st.number_input("Jumlah tablet FE yang diminum selama hamil", min_value = 0, step = 1)
        apakah_ada_komplikasi_kehamilan = st.multiselect("Komplikasi kehamilan", options=list(komplikasi_opt.keys()), format_func=lambda x: komplikasi_opt.get(x))
        tempat_pemeriksaan_kehamilan = st.multiselect("Tempat pemeriksaan kehamilan", options=list(tempatanc_opt.keys()), format_func=lambda x: tempatanc_opt.get(x))
        tm1, tm2, tm3 = st.columns(3)
        frekuensi_anc_tm1 = tm1.number_input("Frekuensi ANC TM1", min_value = 0, step = 1)
        frekuensi_anc_tm2 = tm2.number_input("Frekuensi ANC TM2", min_value = 0, step = 1)
        frekuensi_anc_tm3 = tm3.number_input("Frekuensi ANC TM3", min_value = 0, step = 1)
        # anc_berat = st.selectbox("Apakah diukur berat badan ketika ANC?", ["Iya", "Tidak"], index = 0)
        # anc_tinggi = st.selectbox("Apakah diukur tinggi badan ketika ANC?", ["Iya", "Tidak"], index = 0)
        # anc_td = st.selectbox("Apakah diukur td ketika ANC?", ["Iya", "Tidak"], index = 0)
        # anc_teshb = st.selectbox("Apakah ditest HB ketika ANC?", ["Iya", "Tidak"], index = 0)
        # anc_tfu = st.selectbox("Apakah diukur tfu ketika ANC?", ["Iya", "Tidak"], index = 0)
        # anc_djj = st.selectbox("Apakah diukur detak jantung ketika ANC?", ["Iya", "Tidak"], index = 0)
        # anc_pd = st.selectbox("Apakah diukur pd ketika ANC?", ["Iya", "Tidak"], index = 0)
        st.markdown(""" ###### Pelayanan ANC """)
        berat, td, tfu = st.columns(3)
        tinggi, teshb, djj = st.columns(3)
        pd, panggul, tt = st.columns(3)
        anc_berat = berat.checkbox('Ditimbang')
        anc_tinggi = tinggi.checkbox('Diukur tinggi badan')
        anc_td = td.checkbox("Cek tekanan darah")
        anc_teshb = teshb.checkbox("Cek haemoglobin")
        anc_tfu = tfu.checkbox("Diukur janin")
        anc_djj = djj.checkbox("Cek detak jantung janin")
        anc_pd = pd.checkbox("Pemeriksaan internal")
        anc_panggulluar = panggul.checkbox("Diukur panggul luar")
        imunisasi_tt = tt.checkbox("Imunisasi TT (Tetanus)")
        st.markdown(""" --- """)

        st.markdown(""" #### DATA LINGKUNGAN RUMAH """)
        sumber_air_minum_utama = st.selectbox("Sumber air minum utama", options=list(sumberair_opt.keys()), format_func=lambda x: sumberair_opt.get(x))
        tempat_buang_air_besar = st.selectbox("Tempat buang air besar", options=list(bab_opt.keys()), format_func=lambda x: bab_opt.get(x))
        tempat_pembuangan_limbah = st.selectbox("Tempat pembuangan limbah", options=list(limbah_opt.keys()), format_func=lambda x: limbah_opt.get(x))
        tempat_pembuangan_sampah = st.selectbox("Tempat pembuangan sampah", options=list(sampah_opt.keys()), format_func=lambda x: sampah_opt.get(x))
        
        # submit_button = st.form_submit_button("Prediksi")
        
        if st.button("Submit"):
            import pandas as pd
            #Memasukkan isi form kedalam dataframe
            data = {'tinggi_badan': tinggi_badan,'berat_badan': berat_badan,'berat_badan_bayi': berat_badan_bayi, 'jenis_kelamin': jenis_kelamin,
            'jumlah_anggota_kel': jumlah_anggota_kel, 'usia_makan_pertama_selain_asi': usia_makan_pertama_selain_asi, 'satuan_usia_makan_pertama':satuan_usia_makan_pertama, 'usia_minum_air_pertamakali': usia_minum_air_pertamakali,
            'satuan_usia_minum_air':satuan_usia_minum_air,'apakah_anak_minum_vitA_6bln_terakhir':apakah_anak_minum_vitA_6bln_terakhir,'apakah_anak_imunisasi_bcg':apakah_anak_imunisasi_bcg,
            'apakah_anak_Imunisasi_polio':apakah_anak_Imunisasi_polio, 'apakah_anak_imunisasi_dpt':apakah_anak_imunisasi_dpt, 'apakah_anak_imunisasi_campak':apakah_anak_imunisasi_campak,
            'apakah_anak_imunisasi_hb':apakah_anak_imunisasi_hb, 'tinggi_badan_ayah':tinggi_badan_ayah, 'apakah_bekerja_ayah':apakah_bekerja_ayah, 'pendidikan_ayah':pendidikan_ayah,
            'apakah_kebiasaan_merokok_sampai_sekarang':apakah_kebiasaan_merokok_sampai_sekarang,'tinggi_badan_ibu':tinggi_badan_ibu,'apakah_bekerja_ibu':apakah_bekerja_ibu, 'pendidikan_ibu':pendidikan_ibu,'pengobatan_anemia':pengobatan_anemia,
            'pengobatan_hipertensi':pengobatan_hipertensi,'pengobatan_dm':pengobatan_dm, 'usia_ibu_melahirkan':usia_ibu_melahirkan, 'usia_kehamilan_saat_persalinan':usia_kehamilan_saat_persalinan, 'satuan_usia_kehamilan':satuan_usia_kehamilan,
            'apakah_lahir_kembar':apakah_lahir_kembar, 'persepsi_ibu_bayi_lebih_besar':persepsi_ibu_bayi_lebih_besar, 'apakah_pernah_menyusui':apakah_pernah_menyusui,
            'jumlah_fe_diminum_selama_hamil':jumlah_fe_diminum_selama_hamil, 'apakah_ada_komplikasi_kehamilan':apakah_ada_komplikasi_kehamilan, 'tempat_pemeriksaan_kehamilan':tempat_pemeriksaan_kehamilan,
            'frekuensi_anc_tm1':frekuensi_anc_tm1, 'frekuensi_anc_tm2':frekuensi_anc_tm2, 'frekuensi_anc_tm3':frekuensi_anc_tm3, 'anc_berat':anc_berat, 'anc_tinggi':anc_tinggi, 'anc_td':anc_td,
            'anc_teshb':anc_teshb, 'anc_tfu':anc_tfu, 'anc_djj':anc_djj, 'anc_pd':anc_pd, 'anc_panggulluar':anc_panggulluar, 'imunisasi_tt':imunisasi_tt, 'sumber_air_minum_utama':sumber_air_minum_utama,
            'tempat_buang_air_besar':tempat_buang_air_besar, 'tempat_pembuangan_limbah':tempat_pembuangan_limbah, 'tempat_pembuangan_sampah':tempat_pembuangan_sampah}
            dataframe = dataframe.append(data, ignore_index=True)
            dataframe['umur_sekarang'] = 0
            #convert true false to 1 / 0
            check = ['apakah_anak_imunisasi_bcg','apakah_anak_Imunisasi_polio','apakah_anak_imunisasi_dpt','apakah_anak_imunisasi_campak','apakah_anak_imunisasi_hb',
            'anc_berat', 'anc_tinggi', 'anc_td', 'anc_teshb', 'anc_tfu', 'anc_djj', 'anc_pd', 'anc_panggulluar', 'imunisasi_tt']
            for col in check:
                if (dataframe[col]==0).any() :
                    dataframe[col] = np.NaN
                else :
                    dataframe[col] = 1
            pengobatan =['pengobatan_anemia', 'pengobatan_hipertensi', 'pengobatan_dm']
            for col in pengobatan:
                if (dataframe[col]==0).any() :
                    dataframe[col] = np.NaN
                else :
                    dataframe[col] = "1.0"
            dataframe['apakah_ada_komplikasi_kehamilan'] = dataframe.apakah_ada_komplikasi_kehamilan.apply(lambda x: ''.join([str(i) for i in x]))
            dataframe['tempat_pemeriksaan_kehamilan'] = dataframe.tempat_pemeriksaan_kehamilan.apply(lambda x: ''.join([str(i) for i in x]))
            #Mengatasi outlier data
            #Menyamakan satuan data
            dataframe.loc[dataframe['satuan_usia_kehamilan'] == 5.0, 'usia_kehamilan_saat_persalinan'] = (dataframe['usia_kehamilan_saat_persalinan'] * 4) #mengubah data usia bulan ke minggu
            dataframe.loc[dataframe['satuan_usia_minum_air'] == 5.0, 'usia_minum_air_pertamakali'] = (dataframe['usia_minum_air_pertamakali'] * 30) #mengubah usia dari bulan ke hari
            dataframe.loc[dataframe['satuan_usia_minum_air'] == 4.0, 'usia_minum_air_pertamakali'] = (dataframe['usia_minum_air_pertamakali'] * 7) #mengubah usia dari minggu ke hari
            dataframe.loc[dataframe['satuan_usia_makan_pertama'] == 4.0, 'usia_makan_pertama_selain_asi'] = (dataframe['usia_makan_pertama_selain_asi'] * 7) #mengubah usia dari minggu ke hari
            dataframe.loc[dataframe['satuan_usia_makan_pertama'] == 5.0, 'usia_makan_pertama_selain_asi'] = (dataframe['usia_makan_pertama_selain_asi'] * 30) #mengubah usia dari bulan ke hari
            #Mengatasi data tidak logis
            dataframe.loc[dataframe['tinggi_badan'] == 0, 'tinggi_badan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['tinggi_badan'] > 100, 'tinggi_badan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['tinggi_badan'] < 20, 'tinggi_badan'] = np.NaN #mengubah dataframe tidak logis
            dataframe.loc[dataframe['berat_badan'] == 0, 'berat_badan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['berat_badan'] > 100, 'berat_badan'] = np.NaN
            dataframe.loc[dataframe['tinggi_badan_ayah'] > 200, 'tinggi_badan_ayah'] = np.NaN #mengubah data yang tidak logis
            dataframe.loc[dataframe['tinggi_badan_ayah'] == 0, 'tinggi_badan_ayah'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['tinggi_badan_ayah'] < 100, 'tinggi_badan_ayah'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['tinggi_badan_ibu'] == 0, 'tinggi_badan_ibu'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['tinggi_badan_ibu'] > 200, 'tinggi_badan_ibu'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['tinggi_badan_ibu'] < 100, 'tinggi_badan_ibu'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['usia_ibu_melahirkan'] < 10, 'usia_ibu_melahirkan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['usia_kehamilan_saat_persalinan'] > 50, 'usia_kehamilan_saat_persalinan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['usia_kehamilan_saat_persalinan'] < 20, 'usia_kehamilan_saat_persalinan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['berat_badan_bayi'] == 0, 'berat_badan_bayi'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['berat_badan_bayi'] > 9, 'berat_badan_bayi'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['usia_minum_air_pertamakali'] == 0, 'usia_minum_air_pertamakali'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['usia_makan_pertama_selain_asi'] == 0, 'usia_makan_pertama_selain_asi'] = np.NaN #mengubah data tidak logis
            dataframe['usia_kehamilan_saat_persalinan'] = dataframe['usia_kehamilan_saat_persalinan'].round()
            dataframe['usia_minum_air_pertamakali'] = dataframe['usia_minum_air_pertamakali'].round()
            dataframe['usia_makan_pertama_selain_asi'] = dataframe['usia_makan_pertama_selain_asi'].round()
            #mengatasi data yang tidak diketahui
            dataframe.loc[dataframe['jumlah_anggota_kel'] == 0, 'jumlah_anggota_kel'] = np.NaN
            field_kat = ['jenis_kelamin','apakah_anak_minum_vitA_6bln_terakhir','pendidikan_ayah','apakah_bekerja_ayah','sumber_air_minum_utama', 'tempat_buang_air_besar', 'tempat_pembuangan_limbah', 'tempat_pembuangan_sampah', 
            'apakah_bekerja_ibu','pendidikan_ibu','apakah_lahir_kembar','apakah_pernah_menyusui','persepsi_ibu_bayi_lebih_besar','tempat_pemeriksaan_kehamilan','apakah_ada_komplikasi_kehamilan']
            for fk in field_kat:
                dataframe.loc[dataframe[fk] == 9999, fk] = np.NaN #mengubah data yang tidak diketahui
            field_num = ['tinggi_badan','berat_badan', 'tinggi_badan_ayah','tinggi_badan_ibu','berat_badan_bayi','jumlah_anggota_kel','usia_ibu_melahirkan','usia_makan_pertama_selain_asi', 'frekuensi_anc_tm1','frekuensi_anc_tm2','frekuensi_anc_tm3','jumlah_fe_diminum_selama_hamil','usia_kehamilan_saat_persalinan','usia_minum_air_pertamakali']
            for fn in field_num:
                dataframe.loc[dataframe[fn] == 0, fn] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['pendidikan_ayah'] == 7, 'pendidikan_ayah'] = np.NaN #mengubah data yang tidak menjawab
            dataframe.loc[dataframe['pendidikan_ayah'] == 9, 'pendidikan_ayah'] = np.NaN #mengubah data yang missing
            dataframe.loc[dataframe['pendidikan_ayah'] >= 98, 'pendidikan_ayah'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['apakah_anak_imunisasi_bcg'] == 8, 'apakah_anak_imunisasi_bcg'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['apakah_anak_Imunisasi_polio'] == 8, 'apakah_anak_Imunisasi_polio'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['apakah_anak_imunisasi_dpt'] == 8, 'apakah_anak_imunisasi_dpt'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['apakah_anak_imunisasi_campak'] == 8, 'apakah_anak_imunisasi_campak'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['apakah_anak_imunisasi_hb'] == 8, 'apakah_anak_imunisasi_hb'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['apakah_kebiasaan_merokok_sampai_sekarang'] > 3, 'apakah_kebiasaan_merokok_sampai_sekarang'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['sumber_air_minum_utama'] == 99, 'sumber_air_minum_utama'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['tempat_buang_air_besar'] == 99, 'tempat_buang_air_besar'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['tempat_pembuangan_limbah'] == 99, 'tempat_pembuangan_limbah'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['tempat_pembuangan_sampah'] == 99, 'tempat_pembuangan_sampah'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['pendidikan_ibu'] == 7, 'pendidikan_ibu'] = np.NaN #mengubah data yang tidak menjawab
            dataframe.loc[dataframe['pendidikan_ibu'] == 9, 'pendidikan_ibu'] = np.NaN #mengubah data yang missing
            dataframe.loc[dataframe['pendidikan_ibu'] >= 98, 'pendidikan_ibu'] = np.NaN #mengubah data yang tidak tahu
            dataframe.loc[dataframe['persepsi_ibu_bayi_lebih_besar'] >= 8, 'persepsi_ibu_bayi_lebih_besar'] = np.NaN #mengubah data yang tidak tahu
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '99', 'tempat_pemeriksaan_kehamilan'] = np.nan #mengubah data yang tidak tahu
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == 'Z', 'tempat_pemeriksaan_kehamilan'] = np.NaN #mengubah data yang missing
            dataframe.loc[dataframe['apakah_ada_komplikasi_kehamilan'] == '', 'apakah_ada_komplikasi_kehamilan'] = np.NaN #mengubah data yang missing
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '', 'tempat_pemeriksaan_kehamilan'] = np.NaN #mengubah data yang missing
            dataframe.loc[dataframe['apakah_pernah_menyusui'] >= 8, 'apakah_pernah_menyusui'] = np.NaN #mengubah data yang tidak diketahui & missing
            dataframe.loc[dataframe['anc_berat'] >= 8, 'anc_berat'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_tinggi'] >= 8, 'anc_tinggi'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_td'] >= 8, 'anc_td'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_teshb'] >= 8, 'anc_teshb'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_tfu'] >= 8, 'anc_tfu'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_djj'] >= 8, 'anc_djj'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_pd'] >= 8, 'anc_pd'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_panggulluar'] >= 8, 'anc_panggulluar'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['imunisasi_tt'] >= 8, 'imunisasi_tt'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['satuan_usia_minum_air'] >= 9, 'satuan_usia_minum_air'] = np.nan #mengubah satuan yang tidak diketahui (tidak tahu usianya) menjadi NaN
            dataframe.loc[dataframe['usia_makan_pertama_selain_asi'] >= 70, 'usia_makan_pertama_selain_asi'] = np.nan #mengubah usia yang tidak diketahui menjadi NaN
            dataframe.loc[dataframe['usia_minum_air_pertamakali'] >= 70, 'usia_minum_air_pertamakali'] = np.nan #mengubah usia yang tidak diketahui menjadi NaN
            #mengatasi inkonsistensi data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '1', 'tempat_pemeriksaan_kehamilan'] = 'A' #menyamakan kategori data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '2', 'tempat_pemeriksaan_kehamilan'] = 'B' #menyamakan kategori data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '3', 'tempat_pemeriksaan_kehamilan'] = 'C' #menyamakan kategori data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '4', 'tempat_pemeriksaan_kehamilan'] = 'E' #menyamakan kategori data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '5', 'tempat_pemeriksaan_kehamilan'] = 'F' #menyamakan kategori data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '6', 'tempat_pemeriksaan_kehamilan'] = 'G' #menyamakan kategori data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '7', 'tempat_pemeriksaan_kehamilan'] = 'V' #menyamakan kategori data other
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '8', 'tempat_pemeriksaan_kehamilan'] = 'V' #menyamakan kategori data other
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == 'H', 'tempat_pemeriksaan_kehamilan'] = 'V' #menyamakan kategori data other

            #Mengatasi missing value
            #numerik normal
            num1 = ['tinggi_badan','berat_badan', 'tinggi_badan_ayah','tinggi_badan_ibu','berat_badan_bayi']
            impute_mean = load(open('mean.pkl', 'rb'))
            dataframe[num1] = impute_mean.transform(dataframe[num1])
            #numerik skewed
            num2 = ['jumlah_anggota_kel','usia_ibu_melahirkan','usia_makan_pertama_selain_asi', 'frekuensi_anc_tm1','frekuensi_anc_tm2','frekuensi_anc_tm3','jumlah_fe_diminum_selama_hamil','usia_kehamilan_saat_persalinan','usia_minum_air_pertamakali']
            impute_median = load(open('median.pkl', 'rb'))
            dataframe[num2] = impute_median.transform(dataframe[num2])
            #kategorik
            category = ['jenis_kelamin','apakah_anak_minum_vitA_6bln_terakhir','apakah_anak_imunisasi_bcg','apakah_anak_Imunisasi_polio','apakah_anak_imunisasi_dpt','apakah_anak_imunisasi_campak','apakah_anak_imunisasi_hb',
            'pendidikan_ayah','apakah_bekerja_ayah','apakah_kebiasaan_merokok_sampai_sekarang','sumber_air_minum_utama', 'tempat_buang_air_besar', 'tempat_pembuangan_limbah', 'tempat_pembuangan_sampah', 
            'apakah_bekerja_ibu','pendidikan_ibu','pengobatan_anemia','pengobatan_hipertensi','pengobatan_dm','apakah_lahir_kembar','apakah_pernah_menyusui',
            'persepsi_ibu_bayi_lebih_besar','tempat_pemeriksaan_kehamilan','apakah_ada_komplikasi_kehamilan','anc_berat','anc_tinggi','anc_td','anc_teshb','anc_tfu','anc_djj','anc_pd','anc_panggulluar','imunisasi_tt']
            impute_mode = load(open('mode.pkl', 'rb'))
            dataframe[category] = impute_mode.transform(dataframe[category])

            #One hot encoding
            all_cat_columns = ['jenis_kelamin','apakah_anak_minum_vitA_6bln_terakhir','apakah_anak_imunisasi_bcg','apakah_anak_Imunisasi_polio','apakah_anak_imunisasi_dpt','apakah_anak_imunisasi_campak','apakah_anak_imunisasi_hb','apakah_bekerja_ayah','apakah_kebiasaan_merokok_sampai_sekarang','apakah_bekerja_ibu','pengobatan_anemia','pengobatan_hipertensi','pengobatan_dm','apakah_lahir_kembar','apakah_pernah_menyusui','anc_berat','anc_tinggi','anc_td','anc_teshb','anc_tfu','anc_djj','anc_pd','anc_panggulluar','imunisasi_tt','pendidikan_ayah','sumber_air_minum_utama', 'tempat_buang_air_besar', 'tempat_pembuangan_limbah', 'tempat_pembuangan_sampah', 'pendidikan_ibu','persepsi_ibu_bayi_lebih_besar','tempat_pemeriksaan_kehamilan','apakah_ada_komplikasi_kehamilan']
            cat_columns = ['jenis_kelamin','apakah_anak_minum_vitA_6bln_terakhir','apakah_anak_imunisasi_bcg','apakah_anak_Imunisasi_polio','apakah_anak_imunisasi_dpt','apakah_anak_imunisasi_campak','apakah_anak_imunisasi_hb','apakah_bekerja_ayah','apakah_kebiasaan_merokok_sampai_sekarang','apakah_bekerja_ibu','pengobatan_anemia','pengobatan_hipertensi','pengobatan_dm','apakah_lahir_kembar','apakah_pernah_menyusui','anc_berat','anc_tinggi','anc_td','anc_teshb','anc_tfu','anc_djj','anc_pd','anc_panggulluar','imunisasi_tt','pendidikan_ayah','sumber_air_minum_utama', 'tempat_buang_air_besar', 'tempat_pembuangan_limbah', 'tempat_pembuangan_sampah', 'pendidikan_ibu','persepsi_ibu_bayi_lebih_besar']
            cat_dummies = load(open('list_dummies.pkl', 'rb'))
            processed_columns = load(open('columns.pkl', 'rb'))
            processed_columns.remove('pidlink')
            import pandas as pd
            dataframe = pd.get_dummies(dataframe, prefix_sep="__", columns=cat_columns)
            #encoding multi category
            multi_code = ['tempat_pemeriksaan_kehamilan','apakah_ada_komplikasi_kehamilan']
            for code in multi_code:
                dataframe[code] = dataframe[code].astype(str)
                dataframe[code] = dataframe[code].apply(list)
                OneHt = dataframe[code].str.join('|').str.get_dummies()
                OneHt = OneHt.add_prefix(code+'__')
                dataframe = pd.concat([dataframe,OneHt], axis=1)
            #menghapus kolom yang tidak ada pada data train
            for col in dataframe.columns:
                if ("__" in col) and (col.split("__")[0] in all_cat_columns) and col not in cat_dummies:
                    dataframe.drop(col, axis=1, inplace=True)
            #menambahkan kolom yang tidak ada pada dataframe dan ada pada data train
            for col in cat_dummies:
                if col not in dataframe:
                    dataframe[col] = 0
            dataframe = dataframe[processed_columns]
            # st.write(dataframe)

            #Normalisasi
            scaler = load(open('scaler.pkl', 'rb'))
            dataframe = pd.DataFrame(scaler.transform(dataframe), columns = dataframe.columns)

            #reshape
            dataframe = dataframe.values
            dataframe = np.reshape(dataframe, (dataframe.shape[0],1, dataframe.shape[1]))
            
            # #Load model
            model = load_model('model/model.hdf5')
            # Apply model to make predictions
            # st.write(model.predict(dataframe))
            prediction = (model.predict(dataframe)>=0.5).astype(int)
            if prediction == 0:
                prediction = 'Tidak Stunting'
                st.success("Hasil prediksi : " + prediction )
            else :
                prediction = 'Stunting'
                st.warning("Hasil prediksi : " + prediction )

            # st.success(nama_anak + f"{jenis_kelamin} {format_func(jenis_kelamin)} diprediksi TIDAK STUNTING ketika usia 7 tahun" + "\n" + "Dengan probabilitas stunting : 25%")
            # st.write(dataframe)

    else :
        st.markdown(
        """
        <center><h1 class="title">Prediksi Stunting</h1><center/>
        """, unsafe_allow_html=True)

        with open("template.zip", "rb") as fp:
            st.download_button(
                label="Download template & panduan",
                data=fp,
                file_name="template.zip",
                mime="application/zip"
            )

        uploaded_file = st.file_uploader("Upload File")
        if uploaded_file is not None:
            dataframe = pd.read_excel(uploaded_file)
            st.subheader('File')
            st.write(dataframe)
            st.markdown(""" --- """)

            #Mengatasi outlier
            #Menyamakan satuan data
            dataframe.loc[dataframe['satuan_usia_kehamilan'] == 5.0, 'usia_kehamilan_saat_persalinan'] = (dataframe['usia_kehamilan_saat_persalinan'] * 4) #mengubah data usia bulan ke minggu
            dataframe.loc[dataframe['satuan_usia_minum_air'] == 5.0, 'usia_minum_air_pertamakali'] = (dataframe['usia_minum_air_pertamakali'] * 30) #mengubah usia dari bulan ke hari
            dataframe.loc[dataframe['satuan_usia_minum_air'] == 6.0, 'usia_minum_air_pertamakali'] = (dataframe['usia_minum_air_pertamakali'] * 30) #mengubah usia dari bulan ke hari
            dataframe.loc[dataframe['satuan_usia_minum_air'] == 4.0, 'usia_minum_air_pertamakali'] = (dataframe['usia_minum_air_pertamakali'] * 7) #mengubah usia dari minggu ke hari
            dataframe.loc[dataframe['satuan_usia_minum_air'] == 2.0, 'usia_minum_air_pertamakali'] = (dataframe['usia_minum_air_pertamakali'] * 7) #mengubah usia dari minggu ke hari
            dataframe.loc[dataframe['satuan_usia_makan_pertama'] == 4.0, 'usia_makan_pertama_selain_asi'] = (dataframe['usia_makan_pertama_selain_asi'] * 7) #mengubah usia dari minggu ke hari
            dataframe.loc[dataframe['satuan_usia_makan_pertama'] == 5.0, 'usia_makan_pertama_selain_asi'] = (dataframe['usia_makan_pertama_selain_asi'] * 30) #mengubah usia dari bulan ke hari
            #Mengatasi data tidak logis
            dataframe.loc[dataframe['tinggi_badan'] == 0, 'tinggi_badan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['tinggi_badan'] > 100, 'tinggi_badan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['tinggi_badan'] < 20, 'tinggi_badan'] = np.NaN #mengubah dataframe tidak logis
            dataframe.loc[dataframe['berat_badan'] == 0, 'berat_badan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['berat_badan'] > 20, 'berat_badan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['berat_badan'] < 2, 'berat_badan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['tinggi_badan_ayah'] > 200, 'tinggi_badan_ayah'] = np.NaN #mengubah data yang tidak logis
            dataframe.loc[dataframe['tinggi_badan_ayah'] == 0, 'tinggi_badan_ayah'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['tinggi_badan_ayah'] < 100, 'tinggi_badan_ayah'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['tinggi_badan_ibu'] == 0, 'tinggi_badan_ibu'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['tinggi_badan_ibu'] > 200, 'tinggi_badan_ibu'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['tinggi_badan_ibu'] < 100, 'tinggi_badan_ibu'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['usia_ibu_melahirkan'] < 10, 'usia_ibu_melahirkan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['usia_ibu_melahirkan'] > 100, 'usia_ibu_melahirkan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['usia_kehamilan_saat_persalinan'] > 50, 'usia_kehamilan_saat_persalinan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['usia_kehamilan_saat_persalinan'] < 20, 'usia_kehamilan_saat_persalinan'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['berat_badan_bayi'] == 0, 'berat_badan_bayi'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['berat_badan_bayi'] > 9, 'berat_badan_bayi'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['usia_minum_air_pertamakali'] == 0, 'usia_minum_air_pertamakali'] = np.NaN #mengubah data tidak logis
            dataframe.loc[dataframe['usia_makan_pertama_selain_asi'] == 0, 'usia_makan_pertama_selain_asi'] = np.NaN #mengubah data tidak logis
            dataframe['usia_kehamilan_saat_persalinan'] = dataframe['usia_kehamilan_saat_persalinan'].round()
            dataframe['usia_minum_air_pertamakali'] = dataframe['usia_minum_air_pertamakali'].round()
            dataframe['usia_makan_pertama_selain_asi'] = dataframe['usia_makan_pertama_selain_asi'].round()
            #mengatasi data yang tidak diketahui
            dataframe.loc[dataframe['pendidikan_ayah'] == 7, 'pendidikan_ayah'] = np.NaN #mengubah data yang tidak menjawab
            dataframe.loc[dataframe['pendidikan_ayah'] == 9, 'pendidikan_ayah'] = np.NaN #mengubah data yang missing
            dataframe.loc[dataframe['pendidikan_ayah'] >= 98, 'pendidikan_ayah'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['apakah_anak_imunisasi_bcg'] == 8, 'apakah_anak_imunisasi_bcg'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['apakah_anak_Imunisasi_polio'] == 8, 'apakah_anak_Imunisasi_polio'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['apakah_anak_imunisasi_dpt'] == 8, 'apakah_anak_imunisasi_dpt'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['apakah_anak_imunisasi_campak'] == 8, 'apakah_anak_imunisasi_campak'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['apakah_anak_imunisasi_hb'] == 8, 'apakah_anak_imunisasi_hb'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['apakah_kebiasaan_merokok_sampai_sekarang'] > 3, 'apakah_kebiasaan_merokok_sampai_sekarang'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['sumber_air_minum_utama'] == 99, 'sumber_air_minum_utama'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['tempat_buang_air_besar'] == 99, 'tempat_buang_air_besar'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['tempat_pembuangan_limbah'] == 99, 'tempat_pembuangan_limbah'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['tempat_pembuangan_sampah'] == 99, 'tempat_pembuangan_sampah'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['pendidikan_ibu'] == 7, 'pendidikan_ibu'] = np.NaN #mengubah data yang tidak menjawab
            dataframe.loc[dataframe['pendidikan_ibu'] == 9, 'pendidikan_ibu'] = np.NaN #mengubah data yang missing
            dataframe.loc[dataframe['pendidikan_ibu'] >= 98, 'pendidikan_ibu'] = np.NaN #mengubah data yang tidak tahu
            dataframe.loc[dataframe['persepsi_ibu_bayi_lebih_besar'] >= 8, 'persepsi_ibu_bayi_lebih_besar'] = np.NaN #mengubah data yang tidak tahu
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '99', 'tempat_pemeriksaan_kehamilan'] = np.nan #mengubah data yang tidak tahu
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == 'Z', 'tempat_pemeriksaan_kehamilan'] = np.NaN #mengubah data yang missing
            dataframe.loc[dataframe['apakah_pernah_menyusui'] >= 8, 'apakah_pernah_menyusui'] = np.NaN #mengubah data yang tidak diketahui & missing
            dataframe.loc[dataframe['anc_berat'] >= 8, 'anc_berat'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_tinggi'] >= 8, 'anc_tinggi'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_td'] >= 8, 'anc_td'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_teshb'] >= 8, 'anc_teshb'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_tfu'] >= 8, 'anc_tfu'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_djj'] >= 8, 'anc_djj'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_pd'] >= 8, 'anc_pd'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['anc_panggulluar'] >= 8, 'anc_panggulluar'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['imunisasi_tt'] >= 8, 'imunisasi_tt'] = np.NaN #mengubah data yang tidak diketahui
            dataframe.loc[dataframe['satuan_usia_minum_air'] >= 9, 'satuan_usia_minum_air'] = np.nan #mengubah satuan yang tidak diketahui (tidak tahu usianya) menjadi NaN
            dataframe.loc[dataframe['usia_makan_pertama_selain_asi'] >= 70, 'usia_makan_pertama_selain_asi'] = np.nan #mengubah usia yang tidak diketahui menjadi NaN
            dataframe.loc[dataframe['usia_minum_air_pertamakali'] >= 70, 'usia_minum_air_pertamakali'] = np.nan #mengubah usia yang tidak diketahui menjadi NaN
            #mengatasi inkonsistensi data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '1', 'tempat_pemeriksaan_kehamilan'] = 'A' #menyamakan kategori data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '2', 'tempat_pemeriksaan_kehamilan'] = 'B' #menyamakan kategori data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '3', 'tempat_pemeriksaan_kehamilan'] = 'C' #menyamakan kategori data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '4', 'tempat_pemeriksaan_kehamilan'] = 'E' #menyamakan kategori data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '5', 'tempat_pemeriksaan_kehamilan'] = 'F' #menyamakan kategori data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '6', 'tempat_pemeriksaan_kehamilan'] = 'G' #menyamakan kategori data
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '7', 'tempat_pemeriksaan_kehamilan'] = 'V' #menyamakan kategori data other
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == '8', 'tempat_pemeriksaan_kehamilan'] = 'V' #menyamakan kategori data other
            dataframe.loc[dataframe['tempat_pemeriksaan_kehamilan'] == 'H', 'tempat_pemeriksaan_kehamilan'] = 'V' #menyamakan kategori data other
            
            dataframe['umur_sekarang'] = 0

            #Mengatasi missing value
            #numerik normal
            num1 = ['tinggi_badan','berat_badan', 'tinggi_badan_ayah','tinggi_badan_ibu','berat_badan_bayi']
            impute_mean = load(open('mean.pkl', 'rb'))
            dataframe[num1] = impute_mean.transform(dataframe[num1])
            #numerik skewed
            num2 = ['jumlah_anggota_kel','usia_ibu_melahirkan','usia_makan_pertama_selain_asi', 'frekuensi_anc_tm1','frekuensi_anc_tm2','frekuensi_anc_tm3','jumlah_fe_diminum_selama_hamil','usia_kehamilan_saat_persalinan','usia_minum_air_pertamakali']
            impute_median = load(open('median.pkl', 'rb'))
            dataframe[num2] = impute_median.transform(dataframe[num2])
            #kategorik
            category = ['jenis_kelamin','apakah_anak_minum_vitA_6bln_terakhir','apakah_anak_imunisasi_bcg','apakah_anak_Imunisasi_polio','apakah_anak_imunisasi_dpt','apakah_anak_imunisasi_campak','apakah_anak_imunisasi_hb',
                                      'pendidikan_ayah','apakah_bekerja_ayah','apakah_kebiasaan_merokok_sampai_sekarang','sumber_air_minum_utama', 'tempat_buang_air_besar', 'tempat_pembuangan_limbah', 'tempat_pembuangan_sampah', 
                                      'apakah_bekerja_ibu','pendidikan_ibu','pengobatan_anemia','pengobatan_hipertensi','pengobatan_dm','apakah_lahir_kembar','apakah_pernah_menyusui',
                                      'persepsi_ibu_bayi_lebih_besar','tempat_pemeriksaan_kehamilan','apakah_ada_komplikasi_kehamilan','anc_berat','anc_tinggi','anc_td','anc_teshb','anc_tfu','anc_djj','anc_pd','anc_panggulluar','imunisasi_tt']
            impute_mode = load(open('mode.pkl', 'rb'))
            dataframe[category] = impute_mode.transform(dataframe[category])

            #One hot encoding
            # data_ifls = pd.read_csv('dataset_ifls_cleaned.csv')
            all_cat_columns = ['jenis_kelamin','apakah_anak_minum_vitA_6bln_terakhir','apakah_anak_imunisasi_bcg','apakah_anak_Imunisasi_polio','apakah_anak_imunisasi_dpt','apakah_anak_imunisasi_campak','apakah_anak_imunisasi_hb','apakah_bekerja_ayah','apakah_kebiasaan_merokok_sampai_sekarang','apakah_bekerja_ibu','pengobatan_anemia','pengobatan_hipertensi','pengobatan_dm','apakah_lahir_kembar','apakah_pernah_menyusui','anc_berat','anc_tinggi','anc_td','anc_teshb','anc_tfu','anc_djj','anc_pd','anc_panggulluar','imunisasi_tt','pendidikan_ayah','sumber_air_minum_utama', 'tempat_buang_air_besar', 'tempat_pembuangan_limbah', 'tempat_pembuangan_sampah', 'pendidikan_ibu','persepsi_ibu_bayi_lebih_besar','tempat_pemeriksaan_kehamilan','apakah_ada_komplikasi_kehamilan']
            cat_columns = ['jenis_kelamin','apakah_anak_minum_vitA_6bln_terakhir','apakah_anak_imunisasi_bcg','apakah_anak_Imunisasi_polio','apakah_anak_imunisasi_dpt','apakah_anak_imunisasi_campak','apakah_anak_imunisasi_hb','apakah_bekerja_ayah','apakah_kebiasaan_merokok_sampai_sekarang','apakah_bekerja_ibu','pengobatan_anemia','pengobatan_hipertensi','pengobatan_dm','apakah_lahir_kembar','apakah_pernah_menyusui','anc_berat','anc_tinggi','anc_td','anc_teshb','anc_tfu','anc_djj','anc_pd','anc_panggulluar','imunisasi_tt','pendidikan_ayah','sumber_air_minum_utama', 'tempat_buang_air_besar', 'tempat_pembuangan_limbah', 'tempat_pembuangan_sampah', 'pendidikan_ibu','persepsi_ibu_bayi_lebih_besar']
            cat_dummies = load(open('list_dummies.pkl', 'rb'))
            processed_columns = load(open('columns.pkl', 'rb'))
            dataframe = pd.get_dummies(dataframe, prefix_sep="__", columns=cat_columns)
            #encoding multi category
            multi_code = ['tempat_pemeriksaan_kehamilan','apakah_ada_komplikasi_kehamilan']
            for code in multi_code:
                dataframe[code] = dataframe[code].astype(str)
                dataframe[code] = dataframe[code].apply(list)
                OneHt = dataframe[code].str.join('|').str.get_dummies()
                OneHt = OneHt.add_prefix(code+'__')
                dataframe = pd.concat([dataframe,OneHt], axis=1)
            #menghapus kolom yang tidak ada pada data train
            for col in dataframe.columns:
                if ("__" in col) and (col.split("__")[0] in all_cat_columns) and col not in cat_dummies:
                    dataframe.drop(col, axis=1, inplace=True)
            #menambahkan kolom yang tidak ada pada dataframe dan ada pada data train
            for col in cat_dummies:
                if col not in dataframe:
                    dataframe[col] = 0
            dataframe = dataframe[processed_columns]
            
            #Normalisasi
            scaler = load(open('scaler.pkl', 'rb'))
            X_dataframe = dataframe.drop(['pidlink'], axis=1)
            X_dataframe = pd.DataFrame(scaler.transform(X_dataframe), columns = X_dataframe.columns)

            #reshape
            X_dataframe = X_dataframe.values
            X_dataframe = np.reshape(X_dataframe, (X_dataframe.shape[0],1, X_dataframe.shape[1]))
            
            #Load model
            model = load_model('model/model.hdf5')
            # Apply model to make predictions
            prediction = (model.predict(X_dataframe)>=0.5).astype(int)
            prediction = np.reshape(prediction, (prediction.shape[0]))
            # probabilitas = model.predict(X_dataframe) * 100
            # probabilitas = np.reshape(probabilitas, (probabilitas.shape[0]))
            dataset = pd.read_excel(uploaded_file)
            hasil_pred = pd.DataFrame({'pidlink': dataframe['pidlink'], 'prediksi': prediction})
            hasil_pred.loc[(hasil_pred['prediksi'] == 0), 'prediksi'] = 'Tidak Stunting'
            hasil_pred.loc[(hasil_pred['prediksi'] == 1), 'prediksi'] = 'Stunting'
            hasil = pd.merge(dataset, hasil_pred, how="left", on='pidlink')
            st.subheader('Hasil Prediksi')
            hasil_csv = hasil.to_csv().encode('utf-8')
            st.download_button(
                "Download hasil prediksi",
                hasil_csv,
                "hasil_prediksi.csv",
                "text/csv",
                key='download-csv'
            )
            col1, col2 = st.columns(2)
            with col1:
                st.write(hasil_pred)
            with col2:
                jumlah = hasil["prediksi"].value_counts()
                st.write(jumlah)
                # chart = plt.figure(figsize = (5, 5))
                # sns.countplot(y = "prediksi", data = hasil)
                # st.pyplot(chart)