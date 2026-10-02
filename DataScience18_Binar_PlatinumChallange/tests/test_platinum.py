"""Smoke tests for the Platinum package. Run from the package root: python -m pytest"""
import pytest

from sentiment_pipeline import SentimentModel, clean_text, load_alay


@pytest.fixture(scope="module")
def alay():
    return load_alay()


@pytest.fixture(scope="module")
def model():
    return SentimentModel.load("sentiment_model.pkl")


def test_whole_word_cleaning(alay):
    # the old re.sub('di','',t) bug turned 'sedih' into 'seh'
    assert clean_text("saya sedih dengan kondisi ini", alay) == "saya sedih dengan kondisi ini"
    assert clean_text("ENAK bgt!!! 123", alay) == "enak banget"


def test_negation_is_kept(alay):
    assert "tidak" in clean_text("pelayanan tidak ramah", alay).split()


def test_data_loads_with_three_labels():
    df = SentimentModel.load_data("data/datacsv_tosql1.db")
    assert len(df) > 10_000
    assert set(df["label"]) == {"negative", "neutral", "positive"}


@pytest.mark.parametrize("text,expected", [
    ("pelayanan lambat dan makanan tidak enak", "negative"),
    ("tempatnya nyaman, pelayanannya ramah sekali", "positive"),
])
def test_predict(model, text, expected):
    assert model.predict([text])[0] == expected
