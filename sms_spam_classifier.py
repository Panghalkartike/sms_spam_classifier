import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from io import BytesIO
from urllib.request import urlopen
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# URL of the ZIP file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

# Download and extract the ZIP file
response = urlopen(url)
zip_file = zipfile.ZipFile(BytesIO(response.read()))

# Extract the specific file
with zip_file.open('SMSSpamCollection') as file:
    df = pd.read_csv(file, sep='\t', names=['label', 'message'])

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Data preprocessing
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print("\nNull values in the dataset:")
print(df.isnull().sum())

print("\nLabel counts:")
print(df['label'].value_counts())

print("\nDataset statistics:")
print(df.describe())

# Feature extraction
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf.fit_transform(df['message'])
y = df['label']

print("\nShape of the features (X):")
print(X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nShape of the training set (X_train, X_test):")
print(X_train.shape, X_test.shape)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'\nAccuracy: {accuracy}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(class_report)

# Visualization
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
