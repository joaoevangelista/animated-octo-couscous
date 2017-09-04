import pandas as pd
import numpy as np
import json

df = pd.read_json('./data/train.json')

df.head()

# The ingredients can't be used that way to feed an algorithm.
# Sklearn will help dealing with the text.

with open('./data/train.json') as f:
    train = json.load(f)


cuisine = np.array([item['cuisine'] for item in train])
train_ingredients = ["|".join(item['ingredients']) for item in train]

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


vectzr = CountVectorizer(token_pattern='[^|]+')
labelenc = LabelEncoder()
X = vectzr.fit_transform(train_ingredients)
y = labelenc.fit_transform(cuisine)

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = LinearSVC(multi_class='ovr', C=0.5)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(accuracy_score(y_test, pred))
#print(search.best_params_)

# with MultinomialNB(alpha=0.4) & CountVectorizer = 0.72~0.73
# with MultinomialNB & CountVectorizer = s0.72
# with RandomForestClassifier & CountVectorizer = 0.69
# with GradientBoostingClassifier & CountVectorizer = N/A
# with LGBMClassifier & CountVectorizer = 0.70
# with LinearSVC & CountVectorizer = 0.77

with open('./data/test.json') as f:
    test = json.load(f)
    
igs = ["|".join(item['ingredients']) for item in test]
ids = [item['id'] for item in test]
results = []

for ig in igs:
    X = vectzr.transform([ig])
    results.append(clf.predict(X))

df = pd.DataFrame({'id': ids, 'cuisine': labelenc.inverse_transform(results).flatten()})
df = df[df.columns[::-1]]
df.to_csv('results.csv', index=False, encoding='utf-8')