import pandas as pd
from urllib.parse import urlparse
import whois
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('D:/sms spam/data/malicious_phish.csv', encoding='latin-1')

# Function to extract domain age 
def get_domain_age(url):
    try:
        domain = urlparse(url).netloc
        if not domain:
            return -1

        w = whois.whois(domain)
        creation_date = w.creation_date

        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        if creation_date:
            age = (datetime.now() - creation_date).days
            return age if age > 0 else -1
        return -1
    except Exception as e:
        print(f"Error checking domain age for {url}: {str(e)}")
        return -1

print("Fetching domain ages (this may take time)...")

# Compute domain age with multithreading 
with ThreadPoolExecutor(max_workers=8) as executor:
    df['domain_age'] = list(executor.map(get_domain_age, df['url']))

# Save only required columns
df_out = df[['url', 'domain_age']]
df_out.to_csv('/mnt/d/sms spam/data/url_domain_age_only.csv', index=False)

print("Saved: 'url_domain_age_only.csv'")
