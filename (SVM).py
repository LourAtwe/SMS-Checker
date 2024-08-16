import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
import joblib
import numpy as np
from sklearn.svm import SVC
import string
from imblearn.over_sampling import SMOTE



df = pd.read_csv('LOUR.csv', encoding='latin-1')
df.dropna(subset=['Labels', 'SMSes'], inplace=True)
print(df.shape)
class_counts = df['Labels'].value_counts()
df['SMSes'] = df['SMSes'].str.lower()
X = df['SMSes']
Y = df['Labels']



def tokenize_text(tokens):
    if isinstance(tokens, list):
        text = ' '.join(tokens)  # Concatenate tokens into a single string
        return text
    else:
        return str(tokens)  # Return the input as a string if it's not a list


#df['SMSes'] = df['SMSes'].astype(str)
df['SMSes'] = df['SMSes'].apply(tokenize_text)
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, Y, test_size=0.2, random_state=42, stratify=Y)

# Apply SMOTE to the training data
"""
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)"""
#print(pd.Series(y_train_resampled).value_counts())


"""naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_resampled, y_train_resampled)"""

svm_classifier = SVC(kernel='linear',random_state=42)
svm_classifier.fit(X_train, y_train)
# Evaluate the model
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test dataset: {accuracy * 100}")
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f'F1 Score: {f1:.2f}')
filename = 'finalized1_model.sav'
joblib.dump(svm_classifier, filename)
joblib.dump(vectorizer, 'vectorizer1.joblib')
# Make predictions for new data
new_data = [str(input("Enter your SMS: "))]
new_data_transformed = vectorizer.transform(new_data)
predictions = svm_classifier.predict(new_data_transformed)
print("your Text is :",predictions)
from sklearn.metrics import classification_report

print(classification_report(Y_test, y_pred))
