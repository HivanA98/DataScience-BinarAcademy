# Platinum Challenge — Refactor + Peningkatan Akurasi

Analisis sentimen Bahasa Indonesia (3 kelas: negative / neutral / positive),
dibangun ulang dari proyek Binar DS-18. Fokus paket ini: **refactor**,
**improvisasi**, dan **peningkatan akurasi yang terverifikasi**.

## Hasil (bebas leakage, 5-fold StratifiedKFold, 10.933 baris)

| | Accuracy | Macro-F1 | Waktu latih |
|---|---|---|---|
| Resep lama (MLP default, tanpa leakage) | 0.838 | 0.797 | ~124 dtk |
| **Hasil refactor (LinearSVC balanced)** | **0.886** | **0.857** | **~4 dtk** |

Macro-F1 dipakai sebagai metrik utama karena kelas sangat timpang
(positive 58% / negative 31% / neutral 10%), sehingga accuracy menyesatkan.

Laporan per kelas (out-of-fold, model akhir):

```
              precision    recall  f1-score   support
    negative      0.844     0.843     0.844      3412
     neutral      0.813     0.793     0.803      1138
    positive      0.921     0.926     0.923      6383
    accuracy                          0.886     10933
   macro avg      0.859     0.854     0.857     10933
```

## Apa yang diubah dan kenapa

1. **Hilangkan data leakage.** TF-IDF lama di-`fit` ke seluruh data sebelum
   split. Di sini vectorizer di-`fit` hanya di dalam lipatan training (lewat
   `Pipeline` + `StratifiedKFold`). Angka lama 0.836 turun jadi 0.797 saat
   leakage dibuang — jadi 0.857 yang baru adalah kenaikan yang jujur.
2. **Metrik yang benar.** Macro-F1 + confusion matrix + laporan per kelas,
   bukan accuracy tunggal.
3. **Cleansing diperbaiki.** Bug `re.sub` substring (yang merusak
   `sedih`→`seh`) diganti normalisasi per kata. Dictionary alay dibangun sekali.
4. **Negasi dijaga.** Stopword tidak dibuang (daftar lama menghapus
   `tidak/bukan/jangan` — membalik sentimen).
5. **Ketimpangan kelas ditangani** dengan `class_weight='balanced'` →
   recall neutral naik 0.73→0.79.
6. **Model diganti** dari MLP default ke LinearSVC (lebih akurat & ~30x cepat).

## Isi paket

- `sentiment_pipeline.py` — pipeline final (train / evaluate / predict / save-load)
- `benchmark.py` — mereproduksi semua angka di atas (baseline, ablation, bake-off)
- `sentiment_model.pkl` — model terlatih siap pakai
- `data/` — dataset SQLite + kamus alay + stopword
- `requirements.txt`

## Cara pakai

```bash
pip install -r requirements.txt
python sentiment_pipeline.py --data data/datacsv_tosql1.db --train   # latih + evaluasi
python benchmark.py                                                  # reproduksi tabel

# inference
python -c "from sentiment_pipeline import SentimentModel; \
m=SentimentModel.load('sentiment_model.pkl'); \
print(m.predict(['pelayanannya lambat dan tidak ramah']))"
```

Langkah lanjutan (IndoBERT, penanganan kelas neutral, tuning) ada di
`CATATAN_BELAJAR.md`.
