"""
app.py — Flask + Swagger API for the Gold challenge (refactored).

Endpoints:
    GET  /                 health check
    POST /cleanse          {"text": "..."}            -> cleaned text
    POST /predict          {"text": "..."}            -> label + probabilities
    POST /predict-file     multipart CSV (col 'text') -> per-row predictions

Model artifacts (model_gold.pkl, tfidf_gold.pkl) are produced by train_gold.py.
Run:  python app.py   then open  http://127.0.0.1:5000/docs
"""
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flasgger import Swagger
from cleansing import clean

app = Flask(__name__)
app.config["SWAGGER"] = {"title": "Sentiment API (Gold)", "uiversion": 3,
                         "specs_route": "/docs"}
swagger = Swagger(app)

# load artifacts once at startup
try:
    MODEL = pickle.load(open("model_gold.pkl", "rb"))
    TFIDF = pickle.load(open("tfidf_gold.pkl", "rb"))
except FileNotFoundError:
    MODEL = TFIDF = None   # run train_gold.py first


def _predict(texts):
    cleaned = [clean(t) for t in texts]
    vecs = TFIDF.transform(cleaned)
    labels = MODEL.predict(vecs)
    probs = MODEL.predict_proba(vecs)
    classes = list(MODEL.classes_)
    out = []
    for raw, lab, pr in zip(texts, labels, probs):
        out.append({"text": raw, "sentiment": lab,
                    "confidence": round(float(max(pr)), 4),
                    "proba": {c: round(float(p), 4) for c, p in zip(classes, pr)}})
    return out


@app.get("/")
def health():
    """Health check.
    ---
    responses:
      200: {description: OK}
    """
    ready = MODEL is not None
    return jsonify(status="ok", model_loaded=ready)


@app.post("/cleanse")
def cleanse():
    """Clean a single text.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema: {type: object, properties: {text: {type: string}}}
    responses:
      200: {description: cleaned text}
    """
    text = (request.get_json(force=True) or {}).get("text", "")
    return jsonify(original=text, cleaned=clean(text))


@app.post("/predict")
def predict():
    """Predict sentiment for a single text.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema: {type: object, properties: {text: {type: string}}}
    responses:
      200: {description: prediction}
    """
    if MODEL is None:
        return jsonify(error="model not trained — run train_gold.py"), 503
    text = (request.get_json(force=True) or {}).get("text", "")
    return jsonify(_predict([text])[0])


@app.post("/predict-file")
def predict_file():
    """Predict sentiment for every row in an uploaded CSV (column 'text').
    ---
    consumes: [multipart/form-data]
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
      200: {description: predictions}
    """
    if MODEL is None:
        return jsonify(error="model not trained — run train_gold.py"), 503
    f = request.files.get("file")
    if f is None:
        return jsonify(error="no file"), 400
    df = pd.read_csv(f)
    col = "text" if "text" in df.columns else df.columns[0]
    return jsonify(results=_predict(df[col].astype(str).tolist()))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
