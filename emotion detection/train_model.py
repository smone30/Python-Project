import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib


df = pd.read_csv("emotion_dataset.csv")
df['clean_text'] = df['text'].apply(nfx.remove_special_characters)


X = df['clean_text']
y = df['emotion']

tfidf = TfidfVectorizer()
X_vect = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "emotion_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("✅ Model and vectorizer saved.")
