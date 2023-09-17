import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# read csv
doc = pd.read_csv('amazon.csv')

# drop rows
doc = doc.iloc[:10000]
print(doc)

# tokenize reviews
t = []
for i in range(0, 10000):
    text = re.sub('[^a-zA-Z]', ' ', doc['Text'][i])
    # convert to lowercase
    text = text.lower()
    # delimiter declaration
    text = text.split()
    # stemming and stopwords removal
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(
        stopwords.words('english'))]
    # add text to corpus
    text = ' '.join(text)
    t.append(text)

cv = CountVectorizer(max_features=1500)

X = cv.fit_transform(t)
y = doc.iloc[:, 1].values

# split into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

# random forest classification
rf_classifier = RandomForestClassifier(
    n_estimators=10, criterion='entropy', random_state=0)

rf_classifier.fit(X_train, y_train)

y_pred_rf = rf_classifier.predict(X_test)

accuracy_score(y_test, y_pred_rf)

# print report
print(classification_report(y_test, y_pred_rf))
