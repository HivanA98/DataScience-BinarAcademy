# Platinum Challenge: Refactor + Accuracy Improvement

[![Platinum CI](https://github.com/HivanA98/DataScience-BinarAcademy/actions/workflows/platinum.yml/badge.svg)](https://github.com/HivanA98/DataScience-BinarAcademy/actions/workflows/platinum.yml)

Indonesian sentiment analysis (3 classes: negative / neutral / positive),
rebuilt from the Binar DS-18 project. This package focuses on **refactoring**,
**improvements**, and **verified accuracy gains**.

## Results (leakage-free, 5-fold StratifiedKFold, 10,933 rows)

| | Accuracy | Macro-F1 | Training time |
|---|---|---|---|
| Original recipe (default MLP, no leakage) | 0.838 | 0.797 | ~124 s |
| **Refactored (balanced LinearSVC)** | **0.886** | **0.857** | **~4 s** |

Macro-F1 is the main metric because the classes are heavily imbalanced
(positive 58% / negative 31% / neutral 10%), which makes accuracy misleading.

Per-class report (out-of-fold, final model):

```
              precision    recall  f1-score   support
    negative      0.844     0.843     0.844      3412
     neutral      0.813     0.793     0.803      1138
    positive      0.921     0.926     0.923      6383
    accuracy                          0.886     10933
   macro avg      0.859     0.854     0.857     10933
```

## What changed and why

1. **Data leakage removed.** The old TF-IDF was `fit` on the full dataset
   before splitting. Here the vectorizer is `fit` only inside the training
   folds (via `Pipeline` + `StratifiedKFold`). The old 0.836 dropped to 0.797
   once the leakage was removed, so the new 0.857 is an honest improvement.
2. **Correct metrics.** Macro-F1 + confusion matrix + per-class report,
   not accuracy alone.
3. **Cleansing fixed.** The `re.sub` substring bug (which turned
   `sedih` → `seh`) was replaced with word-level normalization. The slang
   dictionary is built once.
4. **Negation preserved.** Stopwords are not removed (the old list deleted
   `tidak/bukan/jangan`, which flips the sentiment).
5. **Class imbalance handled** with `class_weight='balanced'`, so
   neutral recall rose from 0.73 to 0.79.
6. **Model replaced** from the default MLP with LinearSVC (more accurate and ~30× faster).

## Package contents

- `sentiment_pipeline.py`: final pipeline (train / evaluate / predict / save-load)
- `benchmark.py`: reproduces every number above (baseline, ablation, bake-off)
- `sentiment_model.pkl`: ready-to-use trained model
- `data/`: SQLite dataset, slang dictionary, and stopword list
- `presentation/`: original team presentation, cleansing screenshots, and manual calculation notes
- `requirements.txt`

## Usage

```bash
pip install -r requirements.txt
python sentiment_pipeline.py --data data/datacsv_tosql1.db --train   # train + evaluate
python benchmark.py                                                  # reproduce the tables

# inference
python -c "from sentiment_pipeline import SentimentModel; \
m=SentimentModel.load('sentiment_model.pkl'); \
print(m.predict(['pelayanannya lambat dan tidak ramah']))"
```

## Tests

```bash
pip install pytest
python -m pytest    # cleansing rules, data loading, predictions
```

GitHub Actions (`.github/workflows/platinum.yml`) runs the 5-fold CV, retrains
the model, and runs these tests on every push or PR that touches this folder.
`benchmark.py` can be run on demand from the Actions tab (**Run workflow** →
tick *benchmark*).

Next steps (IndoBERT, neutral-class handling, tuning) are described in
`CATATAN_BELAJAR.md` (in Indonesian).
