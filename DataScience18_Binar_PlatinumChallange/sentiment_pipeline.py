"""
Indonesian Sentiment Analysis — refactored pipeline
Platinum Challenge (Binar DS-18) rebuild.

Key differences vs the original notebooks:
  1. Cleansing bug fixed: slang/stopword removal now operates on whole WORDS,
     not substrings (original re.sub('di','',t) corrupted 'sedih'->'seh',
     'modifikasi'->'mofikasi', etc.).
  2. No stopword removal for sentiment (the supplied list contained negation
     words 'tidak/bukan/jangan/kurang' — removing them flips sentiment).
  3. No data leakage: TF-IDF is fit inside the CV folds / only on the train
     split, never on the full corpus.
  4. Class imbalance handled with class_weight='balanced'
     (positive 58% / negative 31% / neutral 10%).
  5. Model = LinearSVC (faster and more accurate than default MLP on TF-IDF).
  6. Reported metric is macro-F1, not just accuracy.

Usage:
    python sentiment_pipeline.py --data data/datacsv_tosql1.db --train
    from sentiment_pipeline import SentimentModel
    m = SentimentModel.load('sentiment_model.pkl'); m.predict(['pelayanan lambat sekali'])
"""
from __future__ import annotations
import argparse, re, sqlite3, pickle, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, f1_score
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
LABELS = ["negative", "neutral", "positive"]

# --------------------------------------------------------------------------- #
# Text cleansing
# --------------------------------------------------------------------------- #
_URL   = re.compile(r"(https?://\S+|www\.\S+)")
_NONAZ = re.compile(r"[^a-z\s]")          # keep letters + space only (post-lowercase)
_MULTI = re.compile(r"\s+")
_ELONG = re.compile(r"(.)\1{2,}")         # 'baguuuus' -> 'bagus'


def load_alay(path: str = "data/new_kamusalay.csv") -> dict:
    """Slang -> formal dictionary, built ONCE (original rebuilt it per call)."""
    d = pd.read_csv(path, header=None, encoding="latin-1", names=["slang", "formal"])
    return dict(zip(d["slang"].astype(str), d["formal"].astype(str)))


def clean_text(text: str, alay: dict) -> str:
    """Whole-word cleaning. Negation words are intentionally kept."""
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _ELONG.sub(r"\1", text)
    text = _NONAZ.sub(" ", text)
    text = _MULTI.sub(" ", text).strip()
    return " ".join(alay.get(w, w) for w in text.split() if len(w) > 1)


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
class SentimentModel:
    def __init__(self, alay_path: str = "data/new_kamusalay.csv"):
        self.alay = load_alay(alay_path)
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                      max_df=0.9, sublinear_tf=True)),
            ("clf", LinearSVC(class_weight="balanced", C=1.0,
                              random_state=RANDOM_STATE)),
        ])

    # ---- data ----
    @staticmethod
    def load_data(db_path: str, table: str = "Datanew") -> pd.DataFrame:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql(f'SELECT text, label FROM "{table}"', conn)
        return df.dropna(subset=["text"]).drop_duplicates("text").reset_index(drop=True)

    # ---- evaluation (leakage-free) ----
    def cross_validate(self, texts, y, n_splits: int = 5):
        X = [clean_text(t, self.alay) for t in texts]
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"acc": "accuracy",
                   "macro_f1": make_scorer(f1_score, average="macro")}
        res = cross_validate(self.pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        pred = cross_val_predict(self.pipe, X, y, cv=cv, n_jobs=-1)
        print(f"Accuracy : {res['test_acc'].mean():.4f} +/- {res['test_acc'].std():.4f}")
        print(f"Macro-F1 : {res['test_macro_f1'].mean():.4f} +/- {res['test_macro_f1'].std():.4f}\n")
        print(classification_report(y, pred, digits=3))
        print("Confusion matrix (rows=true, cols=pred):", LABELS)
        print(confusion_matrix(y, pred, labels=LABELS))
        return res

    # ---- fit / predict / io ----
    def fit(self, texts, y):
        self.pipe.fit([clean_text(t, self.alay) for t in texts], y)
        return self

    def predict(self, texts):
        return self.pipe.predict([clean_text(t, self.alay) for t in texts])

    def save(self, path: str = "sentiment_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str = "sentiment_model.pkl") -> "SentimentModel":
        with open(path, "rb") as f:
            return _Unpickler(f).load()


class _Unpickler(pickle.Unpickler):
    """Also accepts models saved by `python sentiment_pipeline.py --train` before
    the __main__ fix, whose pickle references __main__.SentimentModel."""
    def find_class(self, module, name):
        if module == "__main__" and name == "SentimentModel":
            return SentimentModel
        return super().find_class(module, name)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datacsv_tosql1.db")
    ap.add_argument("--alay", default="data/new_kamusalay.csv")
    ap.add_argument("--train", action="store_true", help="fit on all data and save model")
    args = ap.parse_args()

    model = SentimentModel(args.alay)
    df = model.load_data(args.data)
    print(f"Loaded {len(df)} rows | class balance:\n{df['label'].value_counts().to_dict()}\n")

    print("=== Leakage-free 5-fold cross-validation ===")
    model.cross_validate(df["text"].values, df["label"].values)

    if args.train:
        model.fit(df["text"].values, df["label"].values).save()
        print("\nSaved -> sentiment_model.pkl")
        for s in ["pelayanan lambat dan makanan tidak enak",
                  "tempatnya nyaman, pelayanannya ramah sekali",
                  "acara akan diadakan besok pagi"]:
            print(f"  {model.predict([s])[0]:9} <- {s}")


if __name__ == "__main__":
    # run via the importable module so the pickle references
    # sentiment_pipeline.SentimentModel, not __main__.SentimentModel
    import sentiment_pipeline
    sentiment_pipeline.main()
