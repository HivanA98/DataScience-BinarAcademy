Project Team for Platinum Challange by Binar Academy
Data Science Wave 18
##
Feature inside:
##
Cleansing:
- Remove chars: Remove a single word, any numberic, any special word
- Stopwords: Made a few word that common show in all label and don't have any tendency
- KamusAlay: To change any informal or typo word in to Formal word
##
Neural Network:

- Extraction:
  - TF-IDF

- Model:
  - Stratify = true
  - Activation = Relu
  - Solver = adam

- KFold Result:
  - T1: Negative: 0.79, Neutral: 0.79, Positive: 0.89, Accuracy: 0.85
  - T2: Negative: 0.77, Neutral: 0.74, Positive: 0.89, Accuracy: 0.84
  - T3: Negative: 0.79, Neutral: 0.81, Positive: 0.90, Accuracy: 0.85
  - T4: Negative: 0.78, Neutral: 0.74, Positive: 0.89, Accuracy: 0.84
  - T5: Negative: 0.78, Neutral: 0.73, Positive: 0.89, Accuracy: 0.84
  - Average Accuracy: 0.8456363636363637
##
LSTM:

- Extraction:
  - Tokenizer
  - Pad Sequence
    
- Model:
  - Model = sequential
  - Activation = softmax
  - Optimizer = adam
    
- KFold Result:
  - T1: Negative: 0.82, Neutral: 0.74, Positive: 0.91, Accuracy: 0.87
  - T2: Negative: 0.82, Neutral: 0.71, Positive: 0.91, Accuracy: 0.87
  - T3: Negative: 0.82, Neutral: 0.76, Positive: 0.91, Accuracy: 0.87
  - T4: Negative: 0.82, Neutral: 0.75, Positive: 0.91, Accuracy: 0.87
  - T5: Negative: 0.81, Neutral: 0.74, Positive: 0.91, Accuracy: 0.86
  - Accuracy:  0.8659350708733425
##
H5 and Model_of_tfidf download file link:
https://drive.google.com/file/d/103DIJAy_yjSv1rUhSQ6p9WvubW0mFXkC/view?usp=sharing

note:
For model H5 put in main folder flaskswagger
For model of tfidf ekstraktion with the folder model_of_tfidf in the main folder flaskswagger