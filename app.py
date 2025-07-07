from flask import Flask, render_template, request
import pandas as pd
import joblib
import re
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import tldextract
from difflib import SequenceMatcher
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Initialize Flask app
app = Flask(__name__)

# Load model and TLD dictionary
phish_model = joblib.load('phishing_model.pkl')
df4 = pd.read_csv("tld_dict.csv")
TLD_PHISHING_RATE = df4.set_index('TLD')['TLD_encoded'].to_dict()


#define feature extraction functions
def fetch_page(url):
    """Fetch page content with realistic user-agent, retries, and timeout."""
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/122.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive'
    }

    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        response = session.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.text
        else:
            print(f"⚠️ Warning: Status {response.status_code} for URL {url}")
    except requests.exceptions.Timeout:
        print(f"⏱️ Timeout: The request for {url} took too long.")
    except requests.RequestException as e:
        print(f"❌ Error fetching page {url}: {e}")
    
    return None


def has_keyword(text, keywords):
    """Check if any keyword is present in text (case-insensitive)."""
    if text:
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)
    return False

def char_probability(url):
    """Estimate character distribution typicality in URL."""
    total = len(url)
    if total == 0:
        return 0
    special_chars = len(re.findall(r'[^a-zA-Z0-9]', url))
    return 1 - (special_chars / total)  # Higher means more normal characters

def char_continuation_rate(url):
    """Heuristic for repeated character sequences."""
    repeats = len(re.findall(r'(.)\1{2,}', url))  # 3+ repeated chars
    total = len(url)
    if total == 0:
        return 0
    return 1 - (repeats / total)

def spacial_char_ratio(url):
    """Ratio of special chars to total in URL."""
    total = len(url)
    specials = len(re.findall(r'[^a-zA-Z0-9]', url))
    if total == 0:
        return 0
    return specials / total

def similarity(a, b):
    """Levenshtein-based similarity ratio."""
    if not a or not b:
        return 0
    return SequenceMatcher(None, a, b).ratio()

def adjusted_similarity(a, b):
    score = similarity(a, b) * 100
    return 100 if score > 70 else score

def is_legit_domain(base_domain, url):
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return 100 if netloc == base_domain.lower() else adjusted_similarity(base_domain, netloc)
    except:
        return 0
    
def extract_lexical_features(url, domain_title='', page_title=''):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    full_url = url.lower()
    features = {
        'URLSimilarityIndex': is_legit_domain(domain, full_url),
        'DomainTitleMatchScore': adjusted_similarity(domain_title.lower(), domain) if domain_title else 0,
        'URLTitleMatchScore': adjusted_similarity(page_title.lower(), full_url) if page_title else 0,
        'URLCharProb': char_probability(full_url),
        'CharContinuationRate': char_continuation_rate(full_url),
        'SpacialCharRatioInURL': spacial_char_ratio(full_url)
    }
    return features


def extract_structural_features(html):
    """Extract HTML-structure-based features from a web page."""
    soup = BeautifulSoup(html, 'html.parser')

    # Detect presence of submit buttons
    has_submit_button = bool(soup.select('input[type="submit"], button[type="submit"], button'))

    # Detect hidden input fields
    has_hidden_fields = bool(soup.find_all('input', type='hidden'))

    # Detect favicon in various formats
    has_favicon = bool(soup.find('link', rel=lambda rel: rel and 'icon' in rel.lower()))

    # Detect viewport meta tag for responsiveness
    is_responsive = bool(soup.find('meta', attrs={'name': 'viewport'}))

    # Detect presence of meta description
    has_description = bool(soup.find('meta', attrs={'name': 'description'}))

    # Detect copyright notice
    footer = soup.find('footer')
    footer_text = footer.get_text(" ", strip=True).lower() if footer else ''
    has_copyright_info = 'copyright' in footer_text or '©' in footer_text

    return {
        'HasSubmitButton': has_submit_button,
        'HasHiddenFields': has_hidden_fields,
        'HasFavicon': has_favicon,
        'IsResponsive': is_responsive,
        'HasDescription': has_description,
        'HasCopyrightInfo': has_copyright_info
    }

# Extended keyword lists
CRYPTO_KEYWORDS = [
    'btc', 'bitcoin', 'eth', 'ethereum', 'crypto', 'blockchain',
    'coin', 'token', 'web3'
]

PAY_KEYWORDS = [
    'pay', 'shop', 'payment', 'checkout', 'card', 'bank',
    'securepay', 'paypal', 'visa', 'mastercard'
]

SOCIAL_KEYWORDS = [
    'facebook', 'twitter', 'instagram', 'linkedin', 'youtube'
]

def has_keyword(text, keywords):
    if not text:
        return False
    text = text.lower()
    return any(re.search(rf'\b{re.escape(kw)}\b', text) for kw in keywords)

def extract_categorical_features(url, html_text):
    """Extract keyword and scheme based binary features."""
    url_lower = url.lower()
    page_text = html_text.lower() if html_text else ''

    features = {
        'HasSocialNet': any(sk in url_lower for sk in SOCIAL_KEYWORDS) or any(sk in page_text for sk in SOCIAL_KEYWORDS),
        'Crypto': has_keyword(url_lower, CRYPTO_KEYWORDS) or has_keyword(page_text, CRYPTO_KEYWORDS),
        'Pay': has_keyword(url_lower, PAY_KEYWORDS) or has_keyword(page_text, PAY_KEYWORDS),
        'IsHTTPS': url_lower.startswith('https://')
    }
    return features

def encode_tld(url):
    """Map TLD to phishing rate or default 0."""
    ext = tldextract.extract(url)
    tld = ext.suffix.lower()
    return TLD_PHISHING_RATE.get(tld, 0.0)

#load overall feature extraction pipline function to accept urls
def extract_features(url):
    """Robust phishing feature extraction that handles network failures on Streamlit Cloud."""
    html = fetch_page(url)
    page_title = ''
    domain_title = ''

    if html:
        try:
            soup = BeautifulSoup(html, 'html.parser')
            page_title = soup.title.string.strip() if soup.title and soup.title.string else ''
            domain_title_tag = soup.find('title')
            domain_title = domain_title_tag.string.strip() if domain_title_tag and domain_title_tag.string else ''
        except Exception as e:
            print(f"⚠️ HTML parsing failed: {e}")
    else:
        print("⚠️ HTML is None, falling back to URL-only features")

    # Lexical features are always safe (URL-based)
    lexical = extract_lexical_features(url, domain_title, page_title)

    # Structural features from HTML (or fallback to all 0s)
    structural = extract_structural_features(html) if html else {
        'HasSubmitButton': 0,
        'HasHiddenFields': 0,
        'HasFavicon': 0,
        'IsResponsive': 0,
        'HasDescription': 0,
        'HasCopyrightInfo': 0
    }

    # Categorical features (uses either page or just URL text)
    categorical = extract_categorical_features(url, html or '')

    # TLD-based phishing likelihood
    tld_encoded = encode_tld(url)

    # Combine all features
    features = {**categorical, **structural, **lexical, 'TLD_encoded': tld_encoded}
    feature_order = [
        'HasSocialNet', 'HasDescription', 'HasCopyrightInfo', 'HasSubmitButton',
        'HasFavicon', 'Pay', 'HasHiddenFields', 'IsResponsive', 'IsHTTPS', 'Crypto',
        'URLSimilarityIndex', 'DomainTitleMatchScore', 'URLTitleMatchScore',
        'URLCharProb', 'CharContinuationRate', 'SpacialCharRatioInURL',
        'TLD_encoded'
    ]
    feature_vector = [features.get(f, 0) for f in feature_order]
    return features, feature_vector

#function to load features into a database because of what model was trained with

def feature_vector_to_DB(features):
    feature_order = [
        'HasSocialNet', 'HasDescription', 'HasCopyrightInfo', 'HasSubmitButton',
        'HasFavicon', 'Pay', 'HasHiddenFields', 'IsResponsive', 'IsHTTPS', 'Crypto',
        'URLSimilarityIndex', 'DomainTitleMatchScore', 'URLTitleMatchScore',
        'URLCharProb', 'CharContinuationRate', 'SpacialCharRatioInURL',
        'TLD_encoded'
    ]
    
    features_df = pd.DataFrame([features], columns=feature_order)
    print(features_df)
    return features_df

#function to load the dataframe into the plk model for prediction
def phish_detection(feature_dataframe):
    # Load the trained model once (consider loading outside this function for efficiency)
    loaded_model = joblib.load('phishing_model.pkl')
    
    # Predict class (0 = phishing, 1 = safe)
    prediction = loaded_model.predict(feature_dataframe)[0]
    
    # Predict probability for class 1 (safe)
    proba = loaded_model.predict_proba(feature_dataframe)[0][1]

    # Display results
    if prediction == 1:
        print()
        print(f"✅ Site is SAFE to use Chance of being legit is {proba*100}% ")
    else:
        print(f"❌ Site is NOT SAFE (Phishing)")
    return prediction, proba

#function to run phishing detection pipeline 
def phish_detect_pipeline(test_url):
    # Extract features from URL
    features = extract_features(test_url)
    
    # Convert feature vector to DataFrame for model input
    feature_df = feature_vector_to_DB(features)
    
    # Run prediction
    prediction, probability = phish_detection(feature_df)
    
@app.route('/')
def index():
    return render_template('landing_page.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    url = request.form['url']
    features, feature_vector = extract_features(url)
    feature_df = feature_vector_to_DB(feature_vector)
    prediction, proba = phish_detection(feature_df)

    confidence = round(proba * 100, 2)
    prediction_text = "Safe" if prediction == 1 else "NOT SAFE (Phishing)"

    radar_keys = ['URLCharProb', 'CharContinuationRate', 'SpacialCharRatioInURL', 'TLD_encoded']
    bar_keys = ['URLSimilarityIndex', 'DomainTitleMatchScore', 'URLTitleMatchScore']
    boolean_keys = [
        'HasSocialNet', 'HasDescription', 'HasCopyrightInfo', 'HasSubmitButton',
        'HasFavicon', 'Pay', 'HasHiddenFields', 'IsResponsive', 'IsHTTPS', 'Crypto'
    ]

    radar_features = {k: features[k] for k in radar_keys}
    numerical_features = {k: features[k] for k in bar_keys}
    boolean_features = {k: features[k] for k in boolean_keys}

    return render_template(
        'result_page.html',
        url=url,
        prediction_text=prediction_text,
        confidence=confidence,
        radar_features=radar_features,
        numerical_features=numerical_features,
        boolean_features=boolean_features
    )

if __name__ == '__main__':
    app.run(debug=True)
