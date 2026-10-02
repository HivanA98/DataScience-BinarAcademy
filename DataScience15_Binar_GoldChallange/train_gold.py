"""
train_gold.py — refactored, leakage-free training for the Gold challenge.

Produces two artifacts used by app.py:
    model_gold.pkl      (LinearSVC calibrated for probabilities)
    tfidf_gold.pkl      (fitted TF-IDF vectorizer)

Unlike the original notebook, TF-IDF is fit ONLY on the training split, and
evaluation uses macro-F1 + a full classification report.
"""
import pickle, sqlite3, warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from cleansing import clean
warnings.filterwarnings("ignore")

RS = 42


def load():
    with sqlite3.connect("data/datacsv_tosql1.db") as c:
        df = pd.read_sql("SELECT text,label FROM Datanew", c)
    return df.dropna(subset=["text"]).drop_duplicates("text").reset_index(drop=True)


def main():
    df = load()
    df["clean"] = df["text"].apply(clean)          # light cleansing, negation-safe
    X, y = df["clean"].values, df["label"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
    Xtr_v = tfidf.fit_transform(Xtr)               # fit on TRAIN only (no leakage)
    Xte_v = tfidf.transform(Xte)

    # CalibratedClassifierCV wraps LinearSVC so we also get predict_proba for the API
    base = LinearSVC(class_weight="balanced", C=1.0, random_state=RS)
    model = CalibratedClassifierCV(base, cv=3).fit(Xtr_v, ytr)

    pred = model.predict(Xte_v)
    print("Hold-out results")
    print(f"  Accuracy : {accuracy_score(yte, pred):.4f}")
    print(f"  Macro-F1 : {f1_score(yte, pred, average='macro'):.4f}\n")
    print(classification_report(yte, pred, digits=3))

    pickle.dump(model, open("model_gold.pkl", "wb"))
    pickle.dump(tfidf, open("tfidf_gold.pkl", "wb"))
    print("Saved model_gold.pkl and tfidf_gold.pkl")


if __name__ == "__main__":
    main()
