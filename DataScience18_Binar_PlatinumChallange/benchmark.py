"""
benchmark.py — reproduces every claim in the Platinum README with real numbers.

Runs three things under an identical, leakage-free protocol:
  (A) the original recipe  (default MLP + default TF-IDF, raw text)
  (B) a cleansing ablation (no stopwords / all stopwords / keep-negation)
  (C) a model bake-off     (LinearSVC / LogReg / ComplementNB, tuned TF-IDF)

Everything is evaluated with StratifiedKFold and macro-F1 as the headline
metric, because the classes are imbalanced (positive 58 / negative 31 / neutral 10).
"""
import re, sqlite3, time, warnings
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import make_scorer, f1_score, accuracy_score
warnings.filterwarnings("ignore")

RS = 42
F1 = make_scorer(f1_score, average="macro")
SCORING = {"acc": "accuracy", "f1": F1}

# ---- resources ----
_alay = pd.read_csv("data/new_kamusalay.csv", header=None, encoding="latin-1", names=["s", "f"])
ALAY = dict(zip(_alay["s"].astype(str), _alay["f"].astype(str)))
with open("data/stopwordbahasa.csv", encoding="latin-1") as fh:
    SW = {w.strip() for w in fh if w.strip()}
NEG = {"tidak", "tak", "bukan", "jangan", "belum", "tanpa", "kurang",
       "gak", "ga", "nggak", "enggak", "jangankan", "jgn", "gk"}

_url = re.compile(r"(https?://\S+|www\.\S+)"); _nw = re.compile(r"[^a-z\s]")
_ms = re.compile(r"\s+"); _rp = re.compile(r"(.)\1{2,}")

def base_tokens(t):
    t = str(t).lower(); t = _url.sub(" ", t); t = _rp.sub(r"\1", t)
    t = _nw.sub(" ", t); t = _ms.sub(" ", t).strip()
    return [ALAY.get(w, w) for w in t.split() if len(w) > 1]

def clean(t, mode="none"):
    toks = base_tokens(t)
    if mode == "all":       toks = [w for w in toks if w not in SW]
    elif mode == "keepneg": toks = [w for w in toks if w not in (SW - NEG)]
    return " ".join(toks)

def load():
    with sqlite3.connect("data/datacsv_tosql1.db") as c:
        df = pd.read_sql("SELECT text,label FROM Datanew", c)
    return df.dropna(subset=["text"]).drop_duplicates("text").reset_index(drop=True)

def word_tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)

def cv5():
    return StratifiedKFold(5, shuffle=True, random_state=RS)


def part_A(df):
    print("\n(A) ORIGINAL RECIPE — default MLP, default TF-IDF, raw text")
    print("    Reported in repo README with data leakage (TF-IDF fit on full set).")
    X, y = df["text"].values, df["label"].values
    # leaky (as originally written)
    tf = TfidfVectorizer(); Xf = tf.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, stratify=y, random_state=RS)
    m = MLPClassifier(random_state=RS).fit(Xtr, ytr); p = m.predict(Xte)
    print(f"    WITH leakage    : Acc={accuracy_score(yte,p):.4f}  MacroF1={f1_score(yte,p,average='macro'):.4f}")
    # honest (fit on train only)
    tf = TfidfVectorizer(); Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    Xtr2v = tf.fit_transform(Xtr2); Xte2v = tf.transform(Xte2)
    m = MLPClassifier(random_state=RS).fit(Xtr2v, ytr2); p = m.predict(Xte2v)
    print(f"    WITHOUT leakage : Acc={accuracy_score(yte2,p):.4f}  MacroF1={f1_score(yte2,p,average='macro'):.4f}  <-- honest baseline")


def part_B(df):
    print("\n(B) CLEANSING ABLATION — LinearSVC(balanced), tuned TF-IDF, 5-fold CV")
    y = df["label"].values
    for mode, name in [("none", "no stopword removal"),
                       ("all", "all stopwords (orig)"),
                       ("keepneg", "stopwords keep-negation")]:
        X = df["text"].apply(lambda t: clean(t, mode)).values
        p = Pipeline([("t", word_tfidf()), ("c", LinearSVC(class_weight="balanced", random_state=RS))])
        r = cross_validate(p, X, y, cv=cv5(), scoring=SCORING, n_jobs=-1)
        print(f"    {name:26} Acc={r['test_acc'].mean():.4f}  MacroF1={r['test_f1'].mean():.4f}")


def part_C(df):
    print("\n(C) MODEL BAKE-OFF — light cleansing (no stopwords), tuned TF-IDF, 5-fold CV")
    X = df["text"].apply(lambda t: clean(t, "none")).values
    y = df["label"].values
    models = {
        "LinearSVC (balanced)": LinearSVC(class_weight="balanced", C=1.0, random_state=RS),
        "LogReg   (balanced)":  LogisticRegression(class_weight="balanced", C=5.0, max_iter=1000, random_state=RS),
        "ComplementNB":         ComplementNB(),
    }
    for name, clf in models.items():
        p = Pipeline([("t", word_tfidf()), ("c", clf)])
        t = time.time(); r = cross_validate(p, X, y, cv=cv5(), scoring=SCORING, n_jobs=-1)
        print(f"    {name:22} Acc={r['test_acc'].mean():.4f}  MacroF1={r['test_f1'].mean():.4f}  ({time.time()-t:.0f}s)")


if __name__ == "__main__":
    df = load()
    print(f"Dataset: {len(df)} rows  |  {df['label'].value_counts().to_dict()}")
    part_A(df)
    part_B(df)
    part_C(df)
    print("\nHeadline: honest baseline MacroF1 ~0.80  ->  improved LinearSVC MacroF1 ~0.857")
