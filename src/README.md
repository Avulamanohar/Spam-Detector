### Step 1: Load the Data

First, you need to load your dataset. You can use libraries like `pandas` to handle the data.

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('sms_spam_collection.csv')  # Replace with your file path
print(data.head())
```

### Step 2: Data Preprocessing

1. **Text Cleaning**: Remove any unwanted characters, numbers, or punctuation.
2. **Lowercasing**: Convert all text to lowercase to ensure uniformity.
3. **Tokenization**: Split the text into individual words (tokens).
4. **Removing Stop Words**: Remove common words that may not contribute to the meaning (e.g., "and", "the").
5. **Stemming/Lemmatization**: Reduce words to their base or root form.

You can use libraries like `nltk` or `spaCy` for text processing.

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if you haven't already
nltk.download('stopwords')

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = text.split()
    # Remove stop words and stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the messages
data['cleaned_text'] = data['message'].apply(preprocess_text)
```

### Step 3: Feature Extraction

Convert the cleaned text into numerical features that can be used by machine learning algorithms. A common method is to use the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_text'])

# Labels
y = data['label']  # Assuming 'label' column contains 'spam' or 'ham'
```

### Step 4: Split the Data

Split the dataset into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 5: Train the Model

You can use various algorithms for classification. A common choice for text classification is the Naive Bayes classifier.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Initialize the model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Step 6: Save the Model (Optional)

You can save the trained model and vectorizer for future use.

```python
import joblib

# Save the model and vectorizer
joblib.dump(model, 'sms_spam_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
```

### Step 7: Load the Model (Optional)

To use the saved model later, you can load it as follows:

```python
# Load the model and vectorizer
model = joblib.load('sms_spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example usage
new_message = "Congratulations! You've won a free ticket!"
cleaned_message = preprocess_text(new_message)
vectorized_message = vectorizer.transform([cleaned_message])
prediction = model.predict(vectorized_message)
print("Prediction:", prediction)
```

### Conclusion

This is a basic workflow for preprocessing SMS data and training a spam detection model. You can experiment with different algorithms, hyperparameters, and preprocessing techniques to improve the model's performance.