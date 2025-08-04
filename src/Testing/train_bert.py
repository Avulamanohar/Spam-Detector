import pandas as pd
import numpy as np
import re
import time
import datetime
from urllib.parse import urlparse
from tld import get_tld
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import whois
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('D:/sms spam/data/malicious_phish.csv', encoding='latin-1')

# Feature Engineering Functions
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    return 1 if match else 0

def abnormal_url(url):
    hostname = urlparse(url).hostname
    return 0 if hostname and hostname in url else 1


def count_dot(url):
    return url.count('.')

def count_www(url):
    return url.count('www')

def count_atrate(url):
    return url.count('@')

def no_of_dir(url):
    return urlparse(url).path.count('/')

def no_of_embed(url):
    return urlparse(url).path.count('//')
def shortening_service(url):
    match = re.search(
        r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
        r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
        r'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
        r'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
        r'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
        r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
        r'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
        r'tr\.im|link\.zip\.net',
        url
    )
    return 1 if match else 0
def count_https(url):
    return url.count('https')

def count_http(url):
    return url.count('http')

def count_per(url):
    return url.count('%')

def count_ques(url):
    return url.count('?')

def count_hyphen(url):
    return url.count('-')

def count_equal(url):
    return url.count('=')

def url_length(url):
    return len(str(url))

def hostname_length(url):
    return len(urlparse(url).netloc)
 
def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr', url)
    return 1 if match else 0

def digit_count(url):
    return sum(1 for i in url if i.isnumeric())

def letter_count(url):
    return sum(1 for i in url if i.isalpha())

def fd_length(url):
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

# Apply feature engineering
df['use_of_ip'] = df['url'].apply(having_ip_address)
df['abnormal_url'] = df['url'].apply(abnormal_url)
df['count.'] = df['url'].apply(count_dot)
df['count-www'] = df['url'].apply(count_www)
df['count@'] = df['url'].apply(count_atrate)
df['count_dir'] = df['url'].apply(no_of_dir)
df['count_embed_domian'] = df['url'].apply(no_of_embed)
df['short_url'] = df['url'].apply(shortening_service)
df['count-https'] = df['url'].apply(count_https)
df['count-http'] = df['url'].apply(count_http)
df['count%'] = df['url'].apply(count_per)
df['count?'] = df['url'].apply(count_ques)
df['count-'] = df['url'].apply(count_hyphen)
df['count='] = df['url'].apply(count_equal)
df['url_length'] = df['url'].apply(url_length)
df['hostname_length'] = df['url'].apply(hostname_length)
df['sus_url'] = df['url'].apply(suspicious_words)
df['count-digits'] = df['url'].apply(digit_count)
df['count-letters'] = df['url'].apply(letter_count)
df['fd_length'] = df['url'].apply(fd_length)
df['tld'] = df['url'].apply(lambda i: get_tld(i, fail_silently=True))
df['tld_length'] = df['tld'].apply(tld_length)

# Encode target variable
lb_make = LabelEncoder()
df["type_code"] = lb_make.fit_transform(df["type"])

# Prepare features and target
X = df[['use_of_ip', 'abnormal_url', 'count.', 'count-www', 'count@',
       'count_dir', 'count_embed_domian', 'short_url', 'count-https',
       'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
       'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
       'count-letters']]
y = df['type_code']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5
)

# Model Training
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
rf.fit(X_train, y_train)

xgb_c = XGBClassifier(n_estimators=100)
xgb_c.fit(X_train, y_train)

lgb = LGBMClassifier(objective='multiclass', boosting_type='gbdt', 
                    n_jobs=5, silent=True, random_state=5)
lgb.fit(X_train, y_train)

# Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, 
                              target_names=['benign', 'defacement', 'phishing', 'malware']))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")

print("Random Forest Performance:")
evaluate_model(rf, X_test, y_test)

print("XGBoost Performance:")
evaluate_model(xgb_c, X_test, y_test)

print("LightGBM Performance:")
evaluate_model(lgb, X_test, y_test)

# Prediction Function
def get_prediction_from_url(test_url):
    features = [
        having_ip_address(test_url),
        abnormal_url(test_url),
        count_dot(test_url),
        count_www(test_url),
        count_atrate(test_url),
        no_of_dir(test_url),
        no_of_embed(test_url),
        shortening_service(test_url),
        count_https(test_url),
        count_http(test_url),
        count_per(test_url),
        count_ques(test_url),
        count_hyphen(test_url),
        count_equal(test_url),
        url_length(test_url),
        hostname_length(test_url),
        suspicious_words(test_url),
        digit_count(test_url),
        letter_count(test_url),
        fd_length(test_url),
        tld_length(get_tld(test_url, fail_silently=True))
    ]
    
    features = np.array(features).reshape(1, -1)
    pred = lgb.predict(features)[0]
    
    return {
        0: "SAFE",
        1: "DEFACEMENT",
        2: "PHISHING",
        3: "MALWARE"
    }.get(pred, "UNKNOWN")

# Test the model
test_urls = [
    'titaniumcorporate.co.za',
    'en.wikipedia.org/wiki/North_Dakota',
    'http://paypal.com.login.secure.verify',
    'https://www.google.com',
    'http://bit.ly/2kjh5s',
    'http://192.168.1.1/login',
    'http://free-gift-card.com/claim'
]

print("\nTesting Model with Sample URLs:")
for url in test_urls:
    print(f"{url} -> {get_prediction_from_url(url)}")