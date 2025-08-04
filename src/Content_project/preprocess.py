import re
import logging
from urllib.parse import urlparse, urlunparse, unquote
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_url(url: str, include_features: bool = True) -> Tuple[str, dict]:
    """Extracts raw URL without modifying or normalizing. Features extracted if requested."""
    if not isinstance(url, str) or not url.strip():
        logging.warning("Invalid or empty URL input")
        return "", {}

   
    features = {}
    if include_features:
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc or parsed.path.split('/')[0]
            features = {
                'domain_length': len(netloc),
                'subdomains': netloc.count('.'),
                'is_ip': 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', netloc) else 0,
                'has_query': 1 if parsed.query else 0
            }
        except Exception as e:
            logging.error(f"URL parsing failed for {url}: {e}")

    return url.strip(), features


def preprocess_text(text: str) -> str:
    """Returns raw text without aggressive preprocessing (numbers and special characters retained)."""
    return str(text) if isinstance(text, str) else ''
