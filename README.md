# 🎓 Data Science @ Binar Academy

> My learning journey as a Data Science student at **Binar Academy**: Indonesian-language NLP, from text cleansing to a production-style sentiment API.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)
![Swagger](https://img.shields.io/badge/Swagger-Docs-85EA2D?logo=swagger&logoColor=black)
![Status](https://img.shields.io/badge/status-refactored-brightgreen)
[![Gold CI](https://github.com/hivanarmadi/DataScience-BinarAcademy/actions/workflows/gold.yml/badge.svg)](https://github.com/hivanarmadi/DataScience-BinarAcademy/actions/workflows/gold.yml)
[![Platinum CI](https://github.com/hivanarmadi/DataScience-BinarAcademy/actions/workflows/platinum.yml/badge.svg)](https://github.com/hivanarmadi/DataScience-BinarAcademy/actions/workflows/platinum.yml)

---

## 📌 Highlights

Both challenges were **rebuilt from scratch** to fix bugs and methodology problems in the original notebooks. All numbers below are **leakage-free** (5-fold Stratified CV, ~11k rows).

| Metric | Original recipe | Refactored | Δ |
|---|:---:|:---:|:---:|
| Accuracy | 0.838 | **0.886** | +4.8 pts |
| Macro-F1 | 0.797 | **0.857** | +6.0 pts |
| Neutral-class recall | 0.73 | **0.79** | +6 pts |
| Training time | ~124 s | **~4 s** | ~30× faster |

> 💡 **Key lesson:** the improvement came from *better methodology*, not a fancier model. Removing data leakage, using the right metric, keeping negation words, and picking a model that suits sparse text mattered more than model complexity.

---

## 🗂️ Projects

### 🥇 Gold Challenge: Text Cleansing + Sentiment API
📁 [`DataScience15_Binar_GoldChallange/`](DataScience15_Binar_GoldChallange/)

An Indonesian text-cleansing module plus a **Flask + Swagger** REST API that serves a trained sentiment model.

- `cleansing.py`: word-level slang normalization, negation-aware stopword removal (optional), and optional Sastrawi stemming
- `train_gold.py`: leakage-free training that saves `model_gold.pkl` and `tfidf_gold.pkl`
- `app.py`: REST API with interactive Swagger docs

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/cleanse` | `{"text": "..."}` → cleaned text |
| `POST` | `/predict` | `{"text": "..."}` → label + probabilities |
| `POST` | `/predict-file` | Upload a CSV (`text` column) → per-row predictions |

**Hold-out result:** Accuracy **0.887** · Macro-F1 **0.854**

📊 Original presentation: [`presentation/`](DataScience15_Binar_GoldChallange/presentation/)

### 💎 Platinum Challenge: Sentiment Analysis with Better Accuracy
📁 [`DataScience18_Binar_PlatinumChallange/`](DataScience18_Binar_PlatinumChallange/)

A 3-class sentiment classifier (negative / neutral / positive) with a reproducible benchmark.

- `sentiment_pipeline.py`: final pipeline (train / evaluate / predict / save-load)
- `benchmark.py`: reproduces every number in this README (baseline, ablation, model bake-off)
- `sentiment_model.pkl`: trained model, ready to use

```
              precision    recall  f1-score   support
    negative      0.844     0.843     0.844      3412
     neutral      0.813     0.793     0.803      1138
    positive      0.921     0.926     0.923      6383
    accuracy                          0.886     10933
   macro avg      0.859     0.854     0.857     10933
```

📊 Original team presentation, cleansing screenshots, and manual calculation notes: [`presentation/`](DataScience18_Binar_PlatinumChallange/presentation/)

---

## 🔧 What Was Fixed

| # | Problem in the original code | Fix |
|---|---|---|
| 1 | **Data leakage:** TF-IDF was fit on the full dataset before splitting | Vectorizer fit only on training folds via `Pipeline` + `StratifiedKFold` |
| 2 | **Substring bug:** `re.sub('di','',text)` turned `sedih` → `seh` | Normalization now works on whole words |
| 3 | **Negation removed:** stopword list deleted `tidak/bukan/jangan`, flipping sentiment | Stopword removal is optional and keeps negation words |
| 4 | **Misleading metric:** accuracy alone on imbalanced data (58% / 31% / 10%) | Macro-F1, confusion matrix, and per-class report |
| 5 | **Class imbalance ignored** | `class_weight='balanced'` |
| 6 | **Slow model:** default MLP (~124 s) | LinearSVC, which is more accurate and ~30× faster |
| 7 | **Slang dictionary rebuilt on every call** | Built once at import |
| 8 | **No random seeds**, so results weren't reproducible | `random_state=42` everywhere |

### Cleansing ablation (LinearSVC, 5-fold CV)

| Strategy | Accuracy | Macro-F1 |
|---|:---:|:---:|
| **No stopword removal** | **0.8862** | **0.8567** |
| Remove all stopwords (old approach) | 0.8476 | 0.8036 |
| Remove stopwords, keep negation | 0.8712 | 0.8302 |

### Model bake-off

| Model | Macro-F1 | Training time |
|---|:---:|:---:|
| MLP default (old recipe, no leakage) | 0.797 | ~124 s |
| ComplementNB | 0.792 | ~2 s |
| Logistic Regression (balanced) | 0.852 | ~6 s |
| **LinearSVC (balanced)** | **0.857** | **~4 s** |

---

## 🚀 Quick Start

```bash
git clone https://github.com/hivanarmadi/DataScience-BinarAcademy.git
cd DataScience-BinarAcademy
```

**Gold (API):**

```bash
cd DataScience15_Binar_GoldChallange
pip install -r requirements.txt
python train_gold.py   # optional: pre-trained artifacts are already included
python app.py          # Swagger UI at http://127.0.0.1:5000/docs
```

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text": "makanan tidak enak sama sekali"}'
```

**Platinum (pipeline + benchmark):**

```bash
cd DataScience18_Binar_PlatinumChallange
pip install -r requirements.txt
python sentiment_pipeline.py --data data/datacsv_tosql1.db --train
python benchmark.py
```

**Tests:** run `pip install pytest && python -m pytest` inside either folder. GitHub Actions retrains each model and runs its tests on every push or PR that touches that folder.

---

## 🧭 Roadmap

- [ ] Fine-tune **IndoBERT** (`indobenchmark/indobert-base-p1`) for context-aware sentiment
- [ ] Targeted error analysis to improve the **neutral** class (weakest, F1 ≈ 0.80)
- [ ] Systematic hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV`
- [ ] Pre-trained word embeddings (fastText / Word2Vec) for the LSTM path
- [ ] Explainability with model coefficients, LIME, or SHAP
- [x] CI: `pytest` cleansing + API tests on GitHub Actions
- [ ] MLOps: model versioning, drift monitoring

Detailed learning notes (in Indonesian) are in each project's `CATATAN_BELAJAR.md`.

---

## 🧰 Tech Stack

`Python` · `pandas` · `NumPy` · `SciPy` · `scikit-learn` · `Flask` · `Flasgger (Swagger)` · `Sastrawi` · `SQLite` · `Regex`

---

## 📜 History

| Version | Date | Notes |
|---|---|---|
| V.0.0.1 | 28 May 2024 | First Platinum commit: TF-IDF vs BoW baselines |
| V.0.0.2 | 29 May 2024 | Cleansing experiments (lowercase, demoji, slang, stopwords) |
| V.0.0.3 | 30 May 2024 | Custom slang dictionary, final cleansing & preprocessing |
| V.0.0.4 | 3 Jun 2024 | Team merge: Neural Network, LSTM, Flask & Swagger ([original team repo](https://github.com/Ridzan12/24001074-18-Team_4-Analisis_Data_berdasarkan_Sentimen-Platinum)) |
| **V.1.0.0** | **Oct 2026** | **Full refactor of Gold & Platinum: leakage-free evaluation, +6 pts Macro-F1, ~30× faster training** |

> The original presentation documents are kept in each project's `presentation/` folder. The original notebooks are still in this repo's git history.

---

<p align="center"><b>Ivan Armadi Hasugian</b> · Binar Academy Data Science</p>
