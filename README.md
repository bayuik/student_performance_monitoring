# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

### Latar Belakang

Jaya Jaya Institut adalah perguruan tinggi yang telah berdiri sejak tahun 2000 dan memiliki reputasi baik dalam mencetak lulusan berkualitas. Namun, tantangan besar yang dihadapi saat ini adalah tingginya angka mahasiswa yang dropout.

Hal ini berdampak negatif pada:
- Citra institusi
- Menurunnya kepercayaan masyarakat
- Pengurangan pendapatan dari sisi akademik dan operasional

Untuk mengatasi permasalahan ini, institusi ingin mengadopsi pendekatan berbasis data guna:
1. Mengidentifikasi mahasiswa yang berisiko dropout secara dini
2. Memungkinkan intervensi tepat waktu
3. Menjaga kualitas pendidikan
4. Mempertahankan reputasi kampus
### Permasalahan Bisnis

1. **Tingginya Angka Dropout**
   Meningkatnya jumlah mahasiswa yang tidak menyelesaikan studi berdampak buruk pada:
   - Daya tarik calon mahasiswa baru
   - Kepercayaan masyarakat terhadap institusi

2. **Tidak Tersedianya Sistem Monitoring**
   Absennya sistem pemantauan performa mahasiswa secara real-time menyebabkan:
   - Kesulitan deteksi dini potensi dropout
   - Keterlambatan dalam memberikan intervensi

3. **Kurangnya Pemahaman atas Faktor Penyebab Dropout**
   Institusi belum memiliki gambaran menyeluruh mengenai:
   - Faktor akademik yang memengaruhi keputusan dropout
   - Faktor non-akademik yang berperan dalam pengambilan keputusan

### Cakupan Proyek

1. **Data Preparation & EDA**
   - Mengolah data historis mahasiswa
   - Mengeksplorasi pola dan faktor risiko dropout

2. **Modeling**
   - Membangun model prediktif menggunakan machine learning
   - Memetakan tingkat risiko dropout per mahasiswa

3. **Evaluasi Model**
   - Mengukur akurasi prediksi model
   - Menganalisis efektivitas model dengan metrik performa

4. **Pengembangan Dashboard**
   - Membuat visualisasi interaktif
   - Menampilkan monitoring real-time:
     * Performa akademik
     * Peringatan risiko dropout

5. **Rekomendasi**
   - Menyusun strategi pencegahan berbasis data
   - Memberikan saran tindakan untuk:
     * Dosen/wali akademik
     * Departemen kemahasiswaan

### Persiapan

Sumber data: [Dicoding - Student Performance](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)
Untuk menjalankan dashboard ini, Anda memerlukan lingkungan pengembangan Python dengan beberapa library yang sudah disiapkan. Berikut adalah langkah-langkah untuk menyiapkan environment:

1. **Clone Repository**

   Pertama, clone repository ini ke komputer lokal Anda.

   ```bash
   git clone https://github.com/bayuik/student_performance_monitoring.git
   cd student_performance_monitoring
   ```

2. **Setup Virtual Environment (venv)**

  Pastikan Python 3.10 atau versi lebih baru sudah terinstall.
  Buat virtual environment dengan perintah:
  ```bash
    python -m venv venv
  ```
3. **Aktifkan Virtual Environtment**
```bash
source venv/bin/activate
```
4. **Install Dependensi**
```bash
pip install -r requirements.txt
```

## Business Dashboard

### Tujuan
Membangun sistem prediksi berbasis machine learning dan dashboard visualisasi untuk membantu Jaya Jaya Institut:
- Mengenali mahasiswa yang berisiko dropout
- Mengambil langkah preventif secara tepat waktu

### Penjelasan Dashboard
Dashboard menyediakan gambaran visual yang komprehensif mengenai faktor-faktor yang mempengaruhi risiko dropout melalui:

- Analisis multidimensi
- Visualisasi interaktif
- Peringatan dini berbasis prediksi

Faktor yang dianalisis meliputi:
1. Aspek demografis
2. Latar belakang pendidikan
3. Kinerja akademik
4. Kondisi sosial-ekonomi

### Fitur Data

#### 1. Demografi dan Sosial
- `Marital_status`: Status perkawinan mahasiswa
- `Gender`: Jenis kelamin
- `Nacionality`: Kewarganegaraan
- `Age_at_enrollment`: Usia saat mendaftar

#### 2. Latar Belakang Pendidikan dan Finansial
- `Mothers_occupation`: Pekerjaan ibu
- `Fathers_occupation`: Pekerjaan ayah
- `Debtor`: Status hutang pendidikan
- `Tuition_fees_up_to_date`: Kelengkapan pembayaran SPP

#### 3. Kinerja Akademik
- `Curricular_units_1st_sem_enrolled`: Unit kurikuler semester 1
- `Curricular_units_2nd_sem_enrolled`: Unit kurikuler semester 2

#### 4. Konteks Ekonomi Makro
- `Unemployment_rate`: Tingkat pengangguran
- `Inflation_rate`: Tingkat inflasi
- `GDP`: Pertumbuhan ekonomi

### Target
- `Status`: Indikator kelulusan (Dropout/Lulus)
  - Variabel target untuk model prediktif
  - Dijadikan acuan evaluasi performa model


## Menjalankan Sistem Machine Learning

Untuk menjalankan aplikasi secara lokal:

1. **Persiapan Direktori**
   Pastikan berada di direktori proyek (yang memuat `app.py`)

2. **Menjalankan Aplikasi**
   Jalankan perintah berikut di terminal:
   ```bash
   streamlit run app.py
   ```
   bisa juga diakses melalui link berikut: [Streamlit Student Performance](https://studentperformancemonitoring.streamlit.app/)

## Conclusion

Berdasarkan analisis data yang komprehensif, dapat disimpulkan bahwa risiko dropout mahasiswa dipengaruhi oleh empat kategori faktor utama:

### 1. Faktor Demografis & Sosial
- **Usia**: Mahasiswa yang lebih muda/matang menunjukkan pola risiko berbeda
- **Status pernikahan**: Mahasiswa yang sudah menikah memiliki risiko lebih tinggi
- **Jenis kelamin**: Perbedaan gender menunjukkan pola risiko yang khas
- **Kewarganegaraan**: Mahasiswa internasional menunjukkan kerentanan berbeda

### 2. Faktor Finansial & Latar Belakang Keluarga
- **Pekerjaan orang tua**: Terkait dengan kemampuan finansial mendukung studi
- **Kondisi ekonomi keluarga**: Penghasilan keluarga berpengaruh signifikan
- **Status pembayaran kuliah**: Keterlambatan pembayaran sebagai indikator awal

### 3. Faktor Kinerja Akademik
- **Beban mata kuliah**: Mahasiswa yang mengambil terlalu banyak SKS berisiko tinggi
- **Prestasi semester awal**: Indikator kuat untuk prediksi kelanjutan studi
- **Konsistensi nilai**: Fluktuasi nilai sebagai sinyal peringatan

### 4. Faktor Kondisi Eksternal
- **Kondisi ekonomi makro**:
  - Tingkat pengangguran di wilayah asal
  - Laju inflasi nasional
  - Pertumbuhan PDB regional

### Rekomendasi Strategis
Pemahaman multidimensi ini memungkinkan institusi untuk:
1. Mengembangkan **sistem peringatan dini** berbasis machine learning
2. Merancang **program intervensi terarah** sesuai profil risiko
3. Menyusun **kebijakan pendukung** berbasis evidence
4. Mengoptimalkan **alokasi sumber daya** untuk program retensi

Dengan pendekatan data-driven ini, institusi dapat mengurangi angka dropout secara sistematis sekaligus meningkatkan kualitas pengalaman belajar mahasiswa.
### Rekomendasi Action Items

#### 1. Intervensi Akademik Dini
- Menyediakan program remedial intensif
- Memberikan bimbingan belajar khusus untuk:
  - Mahasiswa dengan nilai rendah
  - Mahasiswa dengan beban SKS tinggi
- Menerapkan sistem mentor-mentee antar mahasiswa

#### 2. Bantuan Keuangan
- Memperluas cakupan program beasiswa
- Melakukan audit rutin status pembayaran (triwulanan)
- Menyediakan opsi pembayaran fleksibel
- Membuat program work-study untuk mahasiswa ekonomi lemah

#### 3. Konseling & Dukungan Psikologis
- Menyediakan layanan konseling gratis untuk:
  - Mahasiswa dari keluarga rentan ekonomi
  - Mahasiswa dengan masalah sosial/adaptasi
- Membentuk peer support group
- Melatih dosen wali sebagai first responder

#### 4. Peningkatan Fasilitas Kampus
- Mengoptimalkan fasilitas untuk mahasiswa:
  - Ruang belajar 24 jam
  - Area istirahat yang nyaman
  - Fasilitas daycare untuk mahasiswa menikah
- Meningkatkan aksesibilitas bagi difabel

#### 5. Monitoring Proaktif
- Mengimplementasikan sistem early warning:
  - Tracking nilai real-time
  - Alert untuk partisipasi kelas
  - Pemantauan frekuensi perpustakaan
- Membentuk tim respons cepat akademik

#### 6. Keterlibatan Orang Tua
- Membangun portal informasi orang tua
- Mengadakan pertemuan rutin (minimal semesteran)
- Membuat newsletter perkembangan akademik
- Menyediakan hotline konsultasi orang tua

#### 7. Evaluasi Latar Belakang Mahasiswa
- Mengembangkan sistem profiling mahasiswa
- Memetakan kebutuhan berdasarkan:
  - Latar belakang ekonomi
  - Budaya daerah asal
  - Pengalaman pendidikan sebelumnya
- Menyesuaikan kebijakan dengan karakteristik mahasiswa

### Tujuan Akhir Dashboard

Dashboard ini dirancang untuk mencapai tiga tujuan utama:

#### 1. Identifikasi & Monitoring Risiko
- **Deteksi dini** mahasiswa berisiko dropout
- **Pemantauan real-time** perkembangan akademik
- **Sistem peringatan** berbasis threshold yang dapat dikustomisasi

#### 2. Dukungan Pengambilan Keputusan
- **Visualisasi data** interaktif untuk analisis mendalam
- **Laporan prediktif** dengan tingkat akurasi tinggi
- **Tools komparasi** antar jurusan/angkatan

#### 3. Peningkatan Tingkat Kelulusan
- **Mekanisme intervensi** terstruktur:
  - Akademik
  - Finansial
  - Psikologis
- **Tracking efektivitas** program retensi
- **Optimasi alokasi** sumber daya institusi

**Outcome yang Diharapkan:**
Dengan sistem ini, Jaya Jaya Institut akan mampu:
- Meningkatkan **responsivitas** terhadap masalah mahasiswa
- Mengembangkan **strategi proaktif** dalam menjaga kualitas pendidikan
- Mencapai **peningkatan 25-30%** dalam tingkat retensi mahasiswa
- Membangun **reputasi institusi** sebagai perguruan tinggi yang peduli
