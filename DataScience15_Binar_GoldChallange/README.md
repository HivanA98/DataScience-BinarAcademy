# Gold Challenge: Refactor + Improvements

Indonesian text cleansing plus a sentiment API (Flask + Swagger), rebuilt
from the Binar DS-18 project. This package focuses on **refactoring** and
**improving** the code so it is correct, fast, and maintainable, and ships a
working API. (For full accuracy optimization, see the Platinum package.)

## What was fixed

1. **Substring bug removed.** Slang/word normalization works on whole words,
   not `re.sub('di','',text)`, which broke words like `sedih` → `seh`.
2. **The slang dictionary is built once** at import time, not on every
   function call (15k entries × 11k rows was a big waste in the old code).
3. **Negation is preserved.** Stopword removal is optional and keeps
   `tidak/bukan/jangan/kurang` (removing them flips the sentiment).
4. **Stemming is optional and off by default.** Sastrawi takes ~35 minutes for
   11k rows and does not improve accuracy on this dataset (tested: +0.0025 macro-F1).
5. **No data leakage.** TF-IDF is `fit` only on the training split.
6. **The API was cleaned up** into clear endpoints with Swagger documentation.

## Package contents

- `cleansing.py`: cleansing module (word-level, negation-aware, optional stemming)
- `train_gold.py`: trains and saves `model_gold.pkl` & `tfidf_gold.pkl`
- `app.py`: Flask + Swagger API
- `model_gold.pkl`, `tfidf_gold.pkl`: ready-to-use trained artifacts
- `data/`: dataset, slang dictionary, and stopword list
- `presentation/`: original Gold Challenge presentation
- `requirements.txt`

Gold model hold-out result: Accuracy 0.887, Macro-F1 0.854.

## Usage

```bash
pip install -r requirements.txt
python train_gold.py        # produces model_gold.pkl + tfidf_gold.pkl
python app.py               # start the API
# open http://127.0.0.1:5000/docs for the Swagger UI
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| GET | `/` | health check |
| POST | `/cleanse` | `{"text": "..."}` → cleaned text |
| POST | `/predict` | `{"text": "..."}` → label + probabilities |
| POST | `/predict-file` | upload a CSV (`text` column) → per-row predictions |

Example:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "makanan tidak enak sama sekali"}'
# -> {"sentiment": "negative", ...}
```

Full learning notes (in Indonesian) are in `CATATAN_BELAJAR.md`.
