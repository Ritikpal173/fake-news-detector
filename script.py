import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Load dataset (raw string handles Windows paths cleanly)
df = pd.read_csv('news.csv')

# Drop missing values if any exist in text or label
df = df.dropna(subset=['text', 'label'])

# 2. Separate features and target
X = df['text']
y = df['label']

# 3. Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)

# 4. Vectorize text
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf.fit_transform(x_train)
tfidf_test = tfidf.transform(x_test)

# 5. Train Passive-Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter=50, random_state=7)
pac.fit(tfidf_train, y_train)

# 6. Evaluate
y_pred = pac.predict(tfidf_test)

print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))