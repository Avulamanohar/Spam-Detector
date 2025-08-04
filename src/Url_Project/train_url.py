import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from urllib.parse import urlparse, urlunparse, unquote
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier


def preprocess_url(url):
    if not isinstance(url, str):
        return ""

    # Clean leading/trailing punctuation
    url = url.strip().strip('\'"()[]{}<>.,;!?')

    # Normalize obfuscations
    url = url.lower()
    url = url.replace("hxxp", "http").replace("hxxps", "https")
    url = url.replace("[.]", ".").replace("(.)", ".").replace("{.}", ".")
    url = url.replace("[dot]", ".").replace(" dot ", ".")
    url = re.sub(r"\s+dot\s+", ".", url, flags=re.IGNORECASE)

    # Decode percent-encoded characters
    url = unquote(url)

    # Add scheme if missing
    if not re.match(r'^(http|https|ftp)://', url):
        url = "http://" + url


    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = re.sub(r'//+', '/', parsed.path)
        path = re.sub(r'/+$', '', path)
        query = "&".join(sorted(parsed.query.split("&"))) if parsed.query else ""
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return url.lower().strip()




def Load_data(path, text_col, label_col, label_map):
    data = pd.read_csv(path, encoding='latin-1')
    data = data[[text_col, label_col]]
    data.columns = ['URL Text', 'Label']
    data.dropna(inplace=True)
    data['Label'] = data['Label'].map(label_map)
    data.dropna(subset=['Label'], inplace=True)
    data['URL Text'] = data['URL Text'].apply(preprocess_url)
    return data

# Load data
data = Load_data(
    'D:/sms spam/data/malicious_phish.csv',
    'url',
    'type',
    {'benign': 0, 'phishing': 1, 'defacement': 1, 'malware': 1}
)

# Split
X = data['URL Text']
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Optional: SMOTE for balancing
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train_tfidf, y_train)

models = {
    #'Naive Bayes': MultinomialNB(),
    #'Logistic Regression': LogisticRegression(),
    'passive_aggressive': PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    
}
for name, model in models.items():
    print(f"\n Training {name}...")
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test_tfidf)

    print(f" {name} Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")

summary = []
for name, model in models.items():
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    summary.append((name, acc, f1))

# Print neatly
print("\n Summary:")
print(f"{'Model':<25}{'Accuracy':<10}{'F1 Score':<10}")
for name, acc, f1 in summary:
    print(f"{name:<25}{acc:<10.4f}{f1:<10.4f}")


# Save
joblib.dump(model, 'url_model.pkl')
joblib.dump(vectorizer, 'url_vectorizer.pkl')
print("Model trained only on URLs. Saved successfully.")
