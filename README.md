# DataScience-BinarAcademy
 My Learning Journey as Student Data Science at Binar Academy
##
Gold Challange:
- Due to my previous @HIvanA98 account was hacked, I reupload it.
  Final Version feature:
  - "gold draft done cleaning.ipynb" --> Cleansing; 
  - "Gold draft Visualisasi.ipynb" --> Visualizing; 
  - "Draft Flask.py" --> Final Task, included clansing, flask, and swagger
##
Platinum Challange:
- V.0.0.1 First Commit
- V.0.0.2 Cleansing Update
- V.0.0.3 Final Cleansing & Preprocess
- V.0.0.4 Final Version with Team Merge Repository

Project Timeline:
##

V.0.0.1 First Commit
28 Mey 2024
- Making 2 Feature Extraction using TFIDF without Cleansing, and BOW without cleansing. And found and Choose the best accurate with good F1 Score is TFIDF without Cleansing

##

V.0.0.2 Cleansing Update

29 Mey 2024
- I made Cleansing unit test KFold for every each of LowerCase, Demoji, ChangeAlay, RemoveChars, and NLTK. With Result:
 - Not any changes at Lowercase and Demoji
 - NLTK and Stopwords made F1 Score be worsen
 - A small improvment F1 using Removechars, Loop 3 the neutral from 78% -> 79%
 - An improvment and deterioration F1 using Changealay:
    - Loops 1 Negative 77% -> 76%, Neutral 79% -> 78%
    - Loops 2 Neutral 74% ->72%
    - Loops 4 Neutral 76% -> 78%
    - Loops 5 Negative 79% -> 80%
 - With Combinize Removechars and ChangeAlay, there are an improvment F1 score, but there is a big problem because when testing using predict, from Positive become Negative Labels 
array([[0.07515561, 0.06784903, 0.85699536]]) -> array([[0.42839065, 0.17934933, 0.39226002]])
    So finally I decided to use only Removechars for Cleansing. = TFIDF-Cleansing.jpynb
##

V.0.0.3
30 Mey 2024
- There is a total change and the final version for Cleansing and Preprocess:
 - Cleansing now using both RemoveChars and ChangeAlay
 - I made my own Dictonary for ChangeAlay
 - I update Remove chars, So only word is accepted. Any Number, Unique Word, Single Letter wil delete
 - I still not using stopwords, but a words that not have any tendency and always show in all labels will be delete in RemoveChars
 - The reason why I put stopwords word in remove chars is to easy for any update

Result: 
Awesome Improvement of F1 Score: 
    - Negative from 77% to--> 79% T-3, 
    - Neutral 81% from 78% to--> 81% T-3, 
    - Positive 81% from 88% to--> 89% T-1,
    - Accuracy from: 0.8445454545454545 --> 0.8463636363636363.
Note: 
 - The improvement ran without any deterioration of each traning sesion, so it's awesome and perfect improvment.
 ##

 V.0.0.4
 3 June 2024
- Combinations from member team in to one repository,
- Where it's include Neural Network, LSTM, Flask and Swagger, and the PPT for presentation.

The original source is here: https://github.com/Ridzan12/24001074-18-Team_4-Analisis_Data_berdasarkan_Sentimen-Platinum

##
Library:
Pandas, Numpy, Regex, Flask, Flagger, Pickle, SKlearn

Cleansing Step:
Lowercase, Demoji, Change informal to formal word/ChangeAlay, censor
