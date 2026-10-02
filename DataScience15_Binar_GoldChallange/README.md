# Gold Challenge — Refactor + Improvisasi

Cleansing teks Bahasa Indonesia + API sentimen (Flask + Swagger), dibangun ulang
dari proyek Binar DS-18. Fokus paket ini: **refactor** dan **improvisasi** kode —
kode yang benar, cepat, dan bisa dipelihara, plus API yang berjalan. (Untuk
optimasi akurasi penuh, lihat paket Platinum.)

## Apa yang diperbaiki

1. **Bug substring dihilangkan.** Normalisasi slang/kata dilakukan per kata utuh,
   bukan `re.sub('di','',text)` yang merusak kata seperti `sedih`→`seh`.
2. **Dictionary alay dibangun sekali** di saat import, bukan tiap pemanggilan
   fungsi (15k entri × 11k baris = pemborosan besar di kode lama).
3. **Negasi dijaga.** Penghapusan stopword bersifat opsional dan
   mempertahankan `tidak/bukan/jangan/kurang` (kalau dibuang, sentimen terbalik).
4. **Stemming opsional & mati secara default.** Sastrawi ~35 menit untuk 11k
   baris dan tidak menaikkan akurasi di dataset ini (diuji: +0.0025 macro-F1).
5. **Tanpa data leakage.** TF-IDF di-`fit` hanya di split training.
6. **API dirapikan** jadi endpoint yang jelas dengan dokumentasi Swagger.

## Isi paket

- `cleansing.py` — modul cleansing (per kata, negation-aware, stemming opsional)
- `train_gold.py` — latih + simpan `model_gold.pkl` & `tfidf_gold.pkl`
- `app.py` — API Flask + Swagger
- `model_gold.pkl`, `tfidf_gold.pkl` — artefak terlatih siap pakai
- `data/` — dataset + kamus + stopword
- `requirements.txt`

Hasil hold-out model Gold: Accuracy 0.887, Macro-F1 0.854.

## Cara pakai

```bash
pip install -r requirements.txt
python train_gold.py        # menghasilkan model_gold.pkl + tfidf_gold.pkl
python app.py               # jalankan API
# buka http://127.0.0.1:5000/docs  untuk Swagger UI
```

Endpoint:

| Method | Path | Fungsi |
|---|---|---|
| GET | `/` | health check |
| POST | `/cleanse` | `{"text": "..."}` → teks bersih |
| POST | `/predict` | `{"text": "..."}` → label + probabilitas |
| POST | `/predict-file` | upload CSV (kolom `text`) → prediksi per baris |

Contoh:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "makanan tidak enak sama sekali"}'
# -> {"sentiment": "negative", ...}
```

Catatan pembelajaran lengkap ada di `CATATAN_BELAJAR.md`.
