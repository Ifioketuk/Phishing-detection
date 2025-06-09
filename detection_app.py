import streamlit as st
import sklearn
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import tldextract
from difflib import SequenceMatcher
from urllib.parse import urlparse
import joblib
import time
import plotly.express as px
import plotly.graph_objects as go
import matplotlib as plt

st.set_page_config(
    page_title="Bello Phishing Detection",
    layout="centered" ,
    page_icon="🎣",
    initial_sidebar_state="expanded"
)


# Load and cache the model plk file 
@st.cache_data
def load_model():
    return joblib.load('phishing_model.pkl')

phish_model = load_model()

# Load TLD dictionary with caching
@st.cache_data
def load_tld_dict():
    df = pd.read_csv("tld_dict.csv")
    return df

df4 = load_tld_dict()
TLD_PHISHING_RATE = df4.set_index('TLD')['TLD_encoded'].to_dict()

#define feature extraction functions
def fetch_page(url):
    """Fetch page content with user-agent header and timeout."""
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; Bot/1.0)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Warning: Received status code {response.status_code} for URL {url}")
    except requests.RequestException as e:
        print(f"Error fetching page {url}: {e}")
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
    """Complete feature extraction pipeline for a single URL."""
    html = fetch_page(url)
    page_title = ''
    domain_title = ''
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        page_title = soup.title.string if soup.title else ''
        domain_title_tag = soup.find('title')
        domain_title = domain_title_tag.string if domain_title_tag else ''
    lexical = extract_lexical_features(url, domain_title, page_title)
    structural = extract_structural_features(html) if html else {
        'HasSubmitButton':0, 'HasHiddenFields':0, 'HasFavicon':0,
        'IsResponsive':0, 'HasDescription':0, 'HasCopyrightInfo':0
    }
    categorical = extract_categorical_features(url, html)
    tld_encoded = encode_tld(url)
    features = {**categorical, **structural, **lexical, 'TLD_encoded': tld_encoded}
    feature_order = [
        'HasSocialNet', 'HasDescription', 'HasCopyrightInfo', 'HasSubmitButton',
        'HasFavicon', 'Pay', 'HasHiddenFields', 'IsResponsive', 'IsHTTPS', 'Crypto',
        'URLSimilarityIndex', 'DomainTitleMatchScore', 'URLTitleMatchScore',
        'URLCharProb', 'CharContinuationRate', 'SpacialCharRatioInURL',
        'TLD_encoded'
    ]
    feature_vector = [features.get(f, 0) for f in feature_order]
    return feature_vector

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
    


# --- Streamlit UI --- #

st.title("🎣 Robiat Phishing Detection Site")

# Session state to avoid re-streaming bio
if 'bio_streamed' not in st.session_state:
    def stream_data1():
        message = (
            "Welcome to Robiat Bello's Phishing Detection App. "
        )

        for word in message.split(" "):
            yield word + " "
            time.sleep(0.07)

    def stream_data2():
        message = (
            "Paste any URL below to analyze if it's legitimate or a 🐟phishing attempt 🎣. "
        )

        for word in message.split(" "):
            yield word + " "
            time.sleep(0.07)

    def stream_data3():
        message = (
            "Our model uses lexical, structural, and categorical features to predict with high confidence."
        )

        for word in message.split(" "):
            yield word + " "
            time.sleep(0.07)

    st.write_stream(stream_data1)
    st.write_stream(stream_data2)
    st.write_stream(stream_data3)
    st.session_state['bio_streamed'] = True
else:
    st.write("Welcome to Robiat Bello's Phishing Detection App.")
    st.write("Paste any URL below to analyze if it's legitimate or a 🐟phishing attempt 🎣.")
    st.write("Our model uses lexical, structural, and categorical features to predict with high confidence.")


# URL input field
url_input  = st.text_input("🔗 Enter a URL to check:", placeholder="https://example.com")

if st.button("Analyze Site"):
    if url_input:
        with st.spinner("Analyzing website... please wait"):
            time.sleep(1.2)  # Artificial delay for UX
            features = extract_features(url_input)
            feature_df = feature_vector_to_DB(features)
            prediction, probability = phish_detection(feature_df)

            # Display prediction
            if prediction == 1:
                st.success(f"✅ Site is SAFE — Confidence: {probability * 100:.2f}%")
            else:
                st.error(f"❌ Site is PHISHING — Confidence: {(1 - probability) * 100:.2f}%")

            # Visualize boolean features as checkboxes
            st.markdown("### ✅ Boolean Feature Overview")
            bool_features = [
                'HasSocialNet', 'HasDescription', 'HasCopyrightInfo',
                'HasSubmitButton', 'HasFavicon', 'Pay', 'HasHiddenFields',
                'IsResponsive', 'IsHTTPS', 'Crypto'
            ]
            for feat in bool_features:
                st.checkbox(feat, value=bool(feature_df[feat].iloc[0]), disabled=True)

            st.subheader("🔍 Extracted Numerical Feature Breakdown")

            radar_features = ['URLCharProb', 'CharContinuationRate', 'SpacialCharRatioInURL', 'TLD_encoded']

            values = feature_df[radar_features].iloc[0].tolist()

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_features,
                fill='toself',
                name='Feature Values',
                marker=dict(color='mediumseagreen')
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                title="🔎 Key Numerical Feature Radar Plot"
            )

            st.plotly_chart(fig, use_container_width=True)


            feature_plot_df = feature_df.drop(labels=bool_features + radar_features, axis=1)

            # Reshape the DataFrame for Plotly bar plot
            feature_plot_df = feature_plot_df.T.reset_index()
            feature_plot_df.columns = ["Feature", "Value"]

            # Plot using Plotly
            st.subheader("📊 Feature Importance Snapshot")
            fig = px.bar(
                feature_plot_df,
                x="Feature",
                y="Value",
                color="Value",
                color_continuous_scale="blues",
                title="Feature Importance Snapshot"
            )
            st.plotly_chart(fig, use_container_width=True)
            # Display raw feature_df
            st.markdown("### 🧾 Raw Feature Table")
            st.dataframe(feature_df)
    else:
        st.warning("⚠️ Please enter a URL before clicking Analyze.")