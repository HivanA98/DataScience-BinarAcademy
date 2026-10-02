# Catatan Belajar — Analisis Sentimen (Binar DS-18)

Dokumen ini bukan pujian dan bukan ringkasan proyek. Isinya tiga hal: kesalahan
teknis yang benar-benar ada di kode lama, konsep yang belum kamu kuasai saat itu,
dan arah improvisasi berikutnya yang realistis. Semua klaim angka di sini berasal
dari eksperimen yang dijalankan ulang di dataset yang sama (11.000 baris,
`datacsv_tosql1.db`), bukan dari perkiraan.

---

## 1. Kesalahan teknis di kode lama (sudah diperbaiki)

Ini bug konkret, bukan soal selera. Sebagian membuat hasil lama secara diam-diam
salah, bukan sekadar kurang optimal.

**a. Bug substring pada `removechars`.**
Kode lama memakai `re.sub('di','',text)`, `re.sub('nya','',text)`, `re.sub('th','',text)`,
dan sejenisnya. `re.sub` tanpa batas kata menghapus pola itu dari **dalam** kata,
bukan hanya kata berdiri sendiri. Akibatnya:

| Input | Output lama | Masalah |
|---|---|---|
| `sedih` | `seh` | kata sentimen negatif kuat rusak total |
| `modifikasi` | `mofikasi` | "di" di tengah kata ikut terhapus |
| `kondisi` | `konsi` | rusak |
| `putih` | `pu` | "th" + "i" ikut terpotong |

Untuk tugas sentimen, menghancurkan kata seperti "sedih" adalah kerusakan sinyal
yang serius. Perbaikannya: normalisasi dilakukan **per kata utuh** (tokenisasi
dulu, baru ganti), bukan `re.sub` substring.

**b. Fungsi `cleaning()` didefinisikan tapi tidak pernah dipanggil** (di
`TFIDF-Cleansing.ipynb`). TF-IDF di-`fit` langsung ke `df['text']` mentah. Jadi
semua kode cleansing di notebook itu praktis kode mati — model dilatih di teks
yang belum dibersihkan. Ini contoh kenapa penting mengecek bahwa transformasi
benar-benar masuk ke pipeline, bukan hanya terdefinisi.

**c. `text.lower(text)`** — `str.lower()` tidak menerima argumen. Baris ini akan
melempar `TypeError` begitu dieksekusi. Bukti tambahan bahwa jalur `cleaning()`
tidak pernah benar-benar dijalankan di notebook tersebut.

**d. `import sqlite3` tidak ada** padahal `sqlite3.connect(...)` dipakai untuk
menyimpan ke DB. `NameError` menunggu.

**e. `changealay` membangun ulang dictionary alay di setiap pemanggilan.**
`dict(zip(...))` dieksekusi untuk tiap baris. Untuk 11k baris x 15k entri, ini
pemborosan besar. Dictionary cukup dibangun sekali di luar fungsi.

---

## 2. Konsep yang belum dikuasai saat itu

Ini bagian yang lebih penting daripada bug. Bug bisa ditambal; pola pikir yang
salah akan terulang di proyek berikutnya.

**a. Kebocoran data (data leakage) saat evaluasi.**
Kode lama menjalankan `tfidf.fit(seluruh data)` **sebelum** `train_test_split`.
Artinya vocabulary dan bobot IDF ikut belajar dari data uji. Angka yang muncul
jadi lebih tinggi dari kemampuan model sebenarnya. Buktinya di dataset ini:

- Resep lama, **dengan** leakage: Acc 0.864 / Macro-F1 0.836
- Resep lama yang sama, **tanpa** leakage: Acc 0.838 / Macro-F1 **0.797**

Selisih ~0.04 macro-F1 itu murni ilusi dari leakage. Cara benar: bungkus
vectorizer + classifier dalam satu `Pipeline`, lalu `fit` hanya di lipatan
training (otomatis lewat `cross_validate`/`StratifiedKFold`).

**b. Metrik salah untuk data tidak seimbang.**
Distribusi kelas: positive 58%, negative 31%, neutral 10%. Menebak "positive"
terus saja sudah dapat akurasi 58%. Accuracy menyembunyikan kegagalan di kelas
minoritas. Di kode lama, recall neutral cuma ~0.73 padahal accuracy terlihat
"bagus". Metrik yang benar di sini adalah **macro-F1** (rata-rata F1 tiap kelas,
tanpa dibobot ukuran kelas) plus **confusion matrix** dan **classification report
per kelas**.

**c. Penanganan negasi di analisis sentimen.**
Ini kesalahan klasik. Daftar `stopwordbahasa.csv` memuat kata negasi:
`tidak`, `bukan`, `jangan`, `kurang`, `belum`, `tanpa`. Kalau stopword ini dibuang,
"tidak enak" berubah jadi "enak" — sentimennya terbalik. Negasi justru pembawa
makna paling penting dalam sentimen dan tidak boleh dihapus.

**d. Kapan stopword removal & stemming membantu — dan kapan tidak.**
Diuji langsung (LinearSVC, TF-IDF, 5-fold CV, macro-F1):

| Strategi cleansing | Accuracy | Macro-F1 |
|---|---|---|
| Tanpa buang stopword | **0.8862** | **0.8567** |
| Buang semua stopword (gaya lama) | 0.8476 | 0.8036 |
| Buang stopword tapi pertahankan negasi | 0.8712 | 0.8302 |

Kesimpulan yang penting: untuk TF-IDF, bobot IDF **sudah** menekan kata yang
sering muncul secara otomatis. Membuang stopword manual malah lebih banyak
membuang sinyal daripada noise. Intuisi "makin banyak dibersihkan makin bagus"
tidak benar di sini — harus diuji, bukan diasumsikan.

Untuk **stemming** (Sastrawi), diuji pada subset seimbang 3.061 baris: tanpa
stemming macro-F1 0.8025, dengan stemming 0.8050 — selisih +0.0025, praktis nol
dan masih di dalam rentang noise antar-lipatan. Selain itu stemming Sastrawi
~0.2 detik/baris (~35 menit untuk 11k baris). Jadi: biaya komputasi besar untuk
manfaat yang tidak terbukti di dataset ini. Bukan berarti stemming selalu buruk,
tapi di sini ia tidak layak dipakai — dan itu keputusan yang harus berdasar
pengujian, bukan kebiasaan.

**e. Menangani ketidakseimbangan kelas.**
Kode lama tidak melakukan apa pun soal ini selain `stratify` saat split (itu untuk
evaluasi adil, bukan solusi). `MLPClassifier` bahkan tidak punya `class_weight`.
Model linear (SVM/LogReg) dengan `class_weight='balanced'` memberi bobot lebih ke
kelas minoritas. Hasilnya recall neutral naik dari 0.73 ke 0.79 tanpa merusak
kelas lain.

**f. Pemilihan model.**
Untuk teks TF-IDF yang berdimensi tinggi dan jarang (sparse), model linear
(LinearSVC, Logistic Regression) biasanya **mengungguli** MLP default: lebih
akurat, jauh lebih cepat, dan lebih stabil. Perbandingan aktual:

| Model | Macro-F1 | Waktu latih |
|---|---|---|
| MLP default (resep lama, tanpa leakage) | 0.797 | ~124 detik |
| ComplementNB | 0.792 | ~2 detik |
| LogReg (balanced) | 0.852 | ~6 detik |
| **LinearSVC (balanced)** | **0.857** | **~4 detik** |

MLP bukan pilihan salah secara mutlak, tapi memakainya dengan setelan default
untuk masalah ini adalah memilih opsi paling lambat sekaligus kurang akurat.

**g. Reproducibility.**
`train_test_split` tanpa `random_state`, `MLPClassifier` tanpa `random_state`.
Hasil berubah tiap dijalankan, jadi sulit membandingkan eksperimen secara adil.
Semua yang acak harus diberi seed.

---

## 3. Hasil akhir setelah perbaikan

Semua di bawah **bebas leakage**, 5-fold StratifiedKFold, dataset penuh:

- **Accuracy: 0.838 → 0.886** (+4.8 poin)
- **Macro-F1: 0.797 → 0.857** (+6.0 poin)
- **Waktu latih: ~124 dtk → ~4 dtk** (sekitar 30x lebih cepat)
- Recall kelas neutral (minoritas): **0.73 → 0.79**

Yang perlu digarisbawahi: kenaikan ini datang bukan dari model yang lebih rumit,
tapi dari metodologi yang lebih benar (hilangkan leakage, metrik tepat, negasi
dijaga, model & bobot yang sesuai). Kualitas metodologi mengalahkan kerumitan
model.

---

## 4. Yang masih bisa diimprovisasi (arah berikutnya)

Realistis, dari yang paling berdampak:

**a. Fine-tune IndoBERT / model transformer Indonesia.**
Ini lompatan terbesar yang tersisa. Model berbasis TF-IDF mengabaikan urutan kata
dan konteks. Transformer seperti `indobenchmark/indobert-base-p1` menangkap
konteks dan biasanya mencapai ~0.92–0.95 akurasi untuk sentimen Bahasa Indonesia.
Ini standar praktik sekarang; TF-IDF + linear adalah baseline yang kuat, bukan
puncak. Ini juga jembatan langsung ke minat "AI-assisted / NLP modern" di CV-mu.

**b. Perbaiki kelas neutral secara khusus.**
Neutral cuma 10% data dan jadi kelas terlemah (F1 ~0.80, sering tertukar dengan
positive/negative — lihat confusion matrix). Opsi: tambah data neutral,
oversampling terarah, atau kalibrasi threshold. Mulai dari **error analysis**:
baca 30–50 kasus neutral yang salah, cari polanya.

**c. Tuning hyperparameter yang sistematis.**
Selama ini nilai `C`, `ngram_range`, `min_df` dipilih manual. `GridSearchCV` atau
`RandomizedSearchCV` di dalam CV akan memilihnya secara terukur, bukan tebakan.

**d. Word embeddings untuk jalur LSTM.**
LSTM lama memakai tokenizer + embedding acak yang dilatih dari nol pada data
kecil. Memakai embedding terlatih (fastText Indonesia / Word2Vec) sebagai lapisan
awal biasanya menaikkan hasil LSTM secara signifikan.

**e. Explainability.**
Pakai koefisien LinearSVC (atau LIME/SHAP) untuk melihat kata mana yang mendorong
tiap prediksi. Berguna untuk audit model dan untuk menjelaskan hasil ke
stakeholder non-teknis — keterampilan yang langsung relevan dengan latar QA-mu.

**f. Disiplin MLOps.**
Versioning artefak model, pengujian API otomatis (pytest untuk endpoint Flask),
pemantauan drift saat data berubah. Ini titik temu alami antara pengalaman QA-mu
dan peran ML — dan pembeda yang jarang dimiliki data scientist murni.

---

## 5. Satu kalimat untuk diingat

Kesalahan terbesar di proyek lama bukan modelnya, tapi **evaluasinya**: angka yang
dilaporkan lebih tinggi dari kemampuan sebenarnya karena leakage, dan accuracy
menutupi kelemahan di kelas minoritas. Sebelum mengejar model yang lebih canggih,
pastikan cara mengukurnya benar dulu — kalau tidak, kamu mengoptimalkan angka yang
salah.
