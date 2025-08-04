import os
import re
import joblib
import whois
from datetime import datetime
from urllib.parse import urlparse
from functools import lru_cache

BASE_DIR = os.path.dirname(__file__)

# Load model safely
model_path = os.path.join(BASE_DIR, "url_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "url_vectorizer.pkl")

# Configuration
TRUSTED_AGE_THRESHOLD = 180  # Days (6 months)
HIGH_RISK_AGE_THRESHOLD = 7  # Days

# Load model assets
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def extract_urls(text):
    """Extract URLs from text using regex"""
    pattern = re.compile(
        r'(?i)\b('
        r'(?:http|https|hxxp|hxxps|ftp)://[^\s\'"<>]+'
        r'|www\.[^\s\'"<>]+'
        r'|(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?'
        r')\b'
    )
    return pattern.findall(text)

@lru_cache(maxsize=100000)
def get_domain_age(url):
    """Cached WHOIS lookup for domain age"""
    try:
        domain = urlparse(url).netloc.lower()
        if not domain:
            return -1
            
        w = whois.whois(domain)
        if not w.domain_name:
            return 0  # Unregistered domain
            
        creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
        return (datetime.now() - creation_date).days if creation_date else -1
    except Exception:
        return -1  # WHOIS failed

def classify_by_age(age):
    """Determine risk level based on domain age"""
    if age == 0:
        return " spam"
    elif age == -1:
        return "spam"
    elif age < HIGH_RISK_AGE_THRESHOLD:
        return "spam"
    elif age < 30:
        return "MEDIUM RISK (7-30 days)"
    elif age < TRUSTED_AGE_THRESHOLD:
        return "CAUTION (30-180 days)"
    else:
        return " SAFE (>180 days)"
"""def is_typosquatting(domain):
    rDetect domains mimicking trusted brands (e.g., g0ole.com)
    trusted_domains = ["google.com", "apple.com", "microsoft.com", "amazon.com"]  # Add more
    for trusted in trusted_domains:
        # Check for common typos (0 instead of o, missing letters, etc.)
        if trusted.replace('o', '0') in domain or \
           trusted.replace('l', '1') in domain or \
           trusted.replace('e', '3') in domain:
            return True
    return False"""

def is_blocklisted(url):
    """Check URL against threat feeds (e.g., VirusTotal)."""
    # Implement API calls here (e.g., Google Safe Browsing)
    return False  # Placeholder


def classify_url_from_text(text):
    """Classify URLs with strict rules for WHOIS failures and typosquatting"""
    urls = extract_urls(text)
    results = []
    
    for raw_url in urls:
        domain = urlparse(raw_url).netloc.lower()
        age = get_domain_age(raw_url)
        age_risk = classify_by_age(age)
        
        # 1. Auto-flag typosquatting domains (e.g., g0ole.com)
        """if is_typosquatting(domain):
            results.append({
                "url": raw_url,
                "domain_age": age,
                "age_risk": " TYPOSQUATTING",
                "final_verdict": "MALICIOUS (Typosquatting)",
                "override_reason": "Domain mimics a trusted brand"
            })
            continue
            """
        # 2. Strict rule for WHOIS failures (age = -1)
        if age == -1:
            results.append({
                "url": raw_url,
                "domain_age": age,
                "age_risk": age_risk,
                "final_verdict": "MALICIOUS (WHOIS Verification Failed)",
                "override_reason": "Domain age could not be verified"
            })
            continue
            
        # 3. Strict rule for new domains (<7 days)
        if age < HIGH_RISK_AGE_THRESHOLD:
            results.append({
                "url": raw_url,
                "domain_age": age,
                "age_risk": age_risk,
                "final_verdict": "MALICIOUS (High-Risk New Domain)",
                "override_reason": f"Domain age {age} < {HIGH_RISK_AGE_THRESHOLD} days"
            })
            continue
             #Auto-approve old domains (>180 days) UNLESS blocklisted
        if age > TRUSTED_AGE_THRESHOLD:
            if is_blocklisted(raw_url):  # Check real-time threat feeds
                results.append({
                    "url": raw_url,
                    "final_verdict": "MALICIOUS (Blocklisted)",
                    "override_reason": "Domain in threat intelligence database"
                })
            else:
                results.append({
                    "url": raw_url,
                    "final_verdict": "SAFE (Old Domain)",
                    "override_reason": f"Age {age} > {TRUSTED_AGE_THRESHOLD} days"
                })
            continue
        # 4. Only run ML model for borderline cases
        vect = vectorizer.transform([raw_url])
        ml_prediction = model.predict(vect)[0]
        
        results.append({
            "url": raw_url,
            "domain_age": age,
            "age_risk": age_risk,
            "final_verdict": "MALICIOUS" if ml_prediction == 1 else "SAFE",
            "override_reason": None
        })
    
    return results
print(classify_url_from_text("Check this link: https://www.google.com and also https://newdomain.xyz"))

