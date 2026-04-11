import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('data/mood_movies.csv')

FEATURES = ['mood_input','energy','company','length_pref','language_pref']
LABEL    = 'mood_category'

X = df[FEATURES]
y = df[LABEL]

le = LabelEncoder()
y_enc = le.fit_transform(y)

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_enc = oe.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

print("Training Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_acc = accuracy_score(y_test, nb.predict(X_test))
print(f"Naive Bayes Accuracy: {nb_acc:.2%}")

print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
print(f"Logistic Regression Accuracy: {lr_acc:.2%}")
print("\nDetailed Report:")
print(classification_report(y_test, lr.predict(X_test), target_names=le.classes_))

os.makedirs('model', exist_ok=True)
with open('model/mood_classifier.pkl',  'wb') as f: pickle.dump(lr, f)
with open('model/label_encoder.pkl',    'wb') as f: pickle.dump(le, f)
with open('model/ordinal_encoder.pkl',  'wb') as f: pickle.dump(oe, f)

print("\n3 model files saved!")
print(f"Winner: Naive Bayes ({nb_acc:.2%}) vs Logistic Regression ({lr_acc:.2%})")