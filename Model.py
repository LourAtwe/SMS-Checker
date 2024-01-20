from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
import joblib


df = pd.read_csv('spam.csv',encoding='latin1')
X = df['v2']
Y = df['v1']

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

df['v2'] = df['v2'].apply(tokenize_text)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

# Create and fit the model
vectorizer = TfidfVectorizer(lowercase=False)
X_train_transformed = vectorizer.fit_transform(X_train)
model = DecisionTreeClassifier()
model.fit(X_train_transformed, Y_train)

# Evaluate the model
X_test_transformed = vectorizer.transform(X_test)
predictions = model.predict(X_test_transformed)
accuracy = accuracy_score(Y_test, predictions)
print(f"Accuracy on the test dataset: {accuracy * 100}")
filename = 'finalized_model.sav'
joblib.dump(model, filename)
joblib.dump(vectorizer, 'vectorizer.joblib')


new_data = [str(input("Enter your SMS: "))]
new_data_transformed = vectorizer.transform(new_data)
predictions = model.predict(new_data_transformed)
print("your Text is :",predictions)

