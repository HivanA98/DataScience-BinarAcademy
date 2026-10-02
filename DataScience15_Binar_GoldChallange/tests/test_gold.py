"""Smoke tests for the Gold package. Run from the package root: python -m pytest"""
import io

import pytest

from cleansing import clean


# ---- cleansing ----
def test_whole_word_normalization_keeps_inner_substrings():
    # the old re.sub('di','',t) bug turned these into 'seh' / 'mofikasi' / 'konsi'
    assert clean("saya sedih") == "saya sedih"
    assert clean("modifikasi kondisi") == "modifikasi kondisi"


def test_slang_and_noise_are_normalized():
    assert clean("ENAK bgt!!! 123 https://x.co") == "enak banget"
    assert clean("kerennnn") == "keren"


def test_negation_survives_stopword_removal():
    assert "tidak" in clean("makanan tidak enak", remove_stopwords=True).split()
    assert "tidak" not in clean("makanan tidak enak", remove_stopwords=True,
                                keep_negation=False).split()


# ---- API ----
@pytest.fixture(scope="module")
def client():
    import app as app_module
    assert app_module.MODEL is not None, "run train_gold.py first"
    return app_module.app.test_client()


def test_health(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.get_json() == {"status": "ok", "model_loaded": True}


def test_cleanse_endpoint(client):
    r = client.post("/cleanse", json={"text": "Mantap bgt"})
    assert r.status_code == 200
    assert r.get_json()["cleaned"] == "mantap banget"


@pytest.mark.parametrize("text,expected", [
    ("makanan tidak enak sama sekali, pelayanan buruk", "negative"),
    ("tempatnya nyaman dan pelayanannya ramah sekali", "positive"),
])
def test_predict_endpoint(client, text, expected):
    r = client.post("/predict", json={"text": text})
    assert r.status_code == 200
    body = r.get_json()
    assert body["sentiment"] == expected
    assert set(body["proba"]) == {"negative", "neutral", "positive"}
    assert abs(sum(body["proba"].values()) - 1) < 1e-3


def test_predict_file_endpoint(client):
    csv = b"text\nmakanan enak sekali\npelayanan buruk dan lambat\n"
    r = client.post("/predict-file", data={"file": (io.BytesIO(csv), "x.csv")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    assert len(r.get_json()["results"]) == 2


def test_swagger_docs(client):
    assert client.get("/docs").status_code == 200
