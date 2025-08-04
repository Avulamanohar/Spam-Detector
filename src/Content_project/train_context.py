import pandas as pd
import re
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse, urlunparse, unquote
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import PassiveAggressiveClassifier,RidgeClassifier
from sklearn.svm import LinearSVC



from preprocess import preprocess_text


def Load_data(path, text_col, label_col, label_map):
    data = pd.read_csv(path, encoding='latin-1')
    data = data[[text_col, label_col]]
    data.columns = ['Email Text', 'Email Type']
    data.dropna(subset=['Email Text', 'Email Type'], inplace=True)
    data['Email Type'] = data['Email Type'].map(label_map)
    return data


data1 = Load_data('D:/sms spam/data/Phishing_Email.csv', 'Email Text', 'Email Type', {
    'Safe Email': 0,
    'Phishing Email': 1
})
data2 = Load_data('D:/sms spam/data/spam.csv', 'v2', 'v1', {
    'ham': 0,
    'spam': 1
})


Data_set = pd.concat([data1, data2], ignore_index=True)
Data_set['Email Text'] = Data_set['Email Text'].apply(preprocess_text)

X = Data_set['Email Text']
y = Data_set['Email Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled =  ros.fit_resample(X_train_tfidf, y_train)


models = {
    #'Naive Bayes': MultinomialNB(),
    #'Logistic Regression': LogisticRegression(),
     'passive_aggressive': PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    #'Ridge Classifier': RidgeClassifier(),
    #'Linear SVC': LinearSVC(),
    

}

for name, model in models.items():
    print(f"\n Training {name}...")
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test_tfidf)

    print(f" {name} Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")


joblib.dump(model, 'context_model.pkl')
joblib.dump(vectorizer, 'context_vectorizer.pkl')
print(" Model and vectorizer saved successfully.")
