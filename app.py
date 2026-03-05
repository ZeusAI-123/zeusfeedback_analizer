import streamlit as st
import pandas as pd
import re
import os
import requests
from bs4 import BeautifulSoup
import nltk
import openai
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time

# ── Selenium (for Quora) ───────────────────────────────────────────────────────
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# ── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZEUS FEEDBACK– Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0d0d;
    color: #f0ece2;
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: linear-gradient(135deg, #0d0d0d 0%, #111827 100%);
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.2rem;
    background: linear-gradient(90deg, #f9a825, #ff6f00, #e91e63);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}

.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #f9a825;
    margin-bottom: 1rem;
}

.badge-0 { background: rgba(249,168,37,0.15); border: 1px solid #f9a825; color: #f9a825; }
.badge-1 { background: rgba(233,30,99,0.15); border: 1px solid #e91e63; color: #e91e63; }
.badge-2 { background: rgba(0,188,212,0.15); border: 1px solid #00bcd4; color: #00bcd4; }
.badge-3 { background: rgba(76,175,80,0.15); border: 1px solid #4caf50; color: #4caf50; }
.badge-4 { background: rgba(156,39,176,0.15); border: 1px solid #9c27b0; color: #9c27b0; }

.badge {
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
}

.source-tag {
    display: inline-block;
    background: rgba(249,168,37,0.1);
    border: 1px solid rgba(249,168,37,0.4);
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 2px 8px;
    color: #f9a825;
    margin-right: 6px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #888;
    padding: 10px 20px;
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: rgba(249,168,37,0.1) !important;
    color: #f9a825 !important;
    border-bottom: 2px solid #f9a825 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #f9a825, #ff6f00);
    color: #0d0d0d;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(249,168,37,0.4);
}

.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #f0ece2 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stSelectbox div[data-baseweb="select"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
}

[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.4) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

.stProgress > div > div {
    background: linear-gradient(90deg, #f9a825, #ff6f00) !important;
}

.stDataFrame {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    overflow: hidden;
}

hr { border-color: rgba(255,255,255,0.08); }

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetricValue"] {
    color: #f9a825 !important;
    font-family: 'Space Mono', monospace !important;
}

.stSuccess { background: rgba(76,175,80,0.1) !important; border-left: 3px solid #4caf50 !important; }
.stError { background: rgba(244,67,54,0.1) !important; border-left: 3px solid #f44336 !important; }
.stInfo { background: rgba(0,188,212,0.08) !important; border-left: 3px solid #00bcd4 !important; }
.stWarning { background: rgba(255,152,0,0.1) !important; border-left: 3px solid #ff9800 !important; }

.pulse { animation: pulse 2s infinite; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
</style>
""", unsafe_allow_html=True)

# ── NLTK downloads ─────────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
download_nltk()

# ── Constants ──────────────────────────────────────────────────────────────────
TOPIC_LABELS = {
    0: "⚡ Service Quality",
    1: "📦 Product Issues",
    2: "💬 Communication",
    3: "⏱️ Speed & Delays",
    4: "💡 Innovation",
}
TOPIC_COLORS = {0: "badge-0", 1: "badge-1", 2: "badge-2", 3: "badge-3", 4: "badge-4"}
FALLBACK_SUGGESTIONS = {
    0: "Improve response time and staff training to elevate service quality.",
    1: "Address product defects, improve QC, and ensure reliable delivery.",
    2: "Set up clearer support channels and proactive customer updates.",
    3: "Streamline internal workflows and reduce processing bottlenecks.",
    4: "Explore new features or improvements based on recurring feedback themes.",
}

# ── Helper: Preprocess ─────────────────────────────────────────────────────────
def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    try:
        words = word_tokenize(text)
    except Exception:
        words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and len(w) > 2]
    try:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]
    except Exception:
        pass
    return ' '.join(words) if len(words) >= 2 else ''

# ── Helper: OpenAI Suggestions ─────────────────────────────────────────────────
def get_suggestions_openai(feedback_list):
    try:
        prompt = (
            "Analyze the following customer feedback entries. "
            "For EACH one, provide a single concise actionable suggestion (1-2 sentences) "
            "to improve customer experience. Number your responses to match the input.\n\n"
        )
        for i, fb in enumerate(feedback_list, 1):
            prompt += f"{i}. {fb}\n"

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in customer experience improvement."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        content = response["choices"][0]["message"]["content"]
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        parsed = [re.sub(r'^\d+[\.\)]\s*', '', l) for l in lines if l]
        parsed = [p for p in parsed if p]
        suggestions = parsed[:len(feedback_list)]
        while len(suggestions) < len(feedback_list):
            suggestions.append("")
        return suggestions
    except Exception:
        return ["" for _ in feedback_list]

# ── Helper: Quora Scraper via Selenium ────────────────────────────────────────
def scrape_quora_selenium(url):
    """Use Selenium headless Chrome to scrape Quora answers."""
    feedbacks = []
    error = None
    driver = None
    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("--window-size=1920,1080")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        # Mask selenium detection
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        driver.get(url)

        # Wait for answer content to load
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".q-text, .spacing_log_answer_content, [class*='answer']"))
            )
        except Exception:
            pass  # continue even if timeout, page might still have content

        # Scroll down to load more answers
        for _ in range(3):
            driver.execute_script("window.scrollBy(0, 1500);")
            time.sleep(1.5)

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Try multiple Quora selectors
        selectors = [
            {'class': re.compile(r'q-text|answer_text|spacing_log_answer')},
        ]
        blocks = []
        for sel in selectors:
            blocks += soup.find_all(['div', 'span', 'p'], sel)

        # Fallback: grab all long paragraphs
        if not blocks:
            blocks = soup.find_all(['p', 'div', 'span'])

        seen, unique = set(), []
        for b in blocks:
            text = b.get_text(separator=' ', strip=True)
            if len(text) > 80 and text not in seen:
                seen.add(text)
                unique.append(text[:500])

        feedbacks = unique[:60]

        if not feedbacks:
            error = "⚠️ Quora loaded but no answer text found. The page may require login to view full answers."

    except Exception as e:
        error = f"❌ Selenium error: {str(e)}"
    finally:
        if driver:
            driver.quit()

    return feedbacks, error

# ── Helper: URL Scraper ────────────────────────────────────────────────────────
def scrape_url(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
    feedbacks, source, error = [], "URL", None
    domain = re.sub(r'https?://(www\.)?', '', url).split('/')[0].lower()

    # ── Quora: use Selenium ────────────────────────────────────────────────────
    if 'quora' in domain:
        source = "Quora"
        if not SELENIUM_AVAILABLE:
            error = "❌ Selenium not installed. Run: pip install selenium webdriver-manager"
            return feedbacks, source, error
        feedbacks, error = scrape_quora_selenium(url)
        return feedbacks, source, error

    # ── All other sites: use requests ─────────────────────────────────────────
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        if False:  # placeholder to keep elif chain intact
            pass
        elif 'reddit' in domain:
            source = "Reddit"
            blocks = soup.find_all(['p', 'div'], class_=re.compile(r'comment|usertext|entry'))
        elif 'trustpilot' in domain:
            source = "Trustpilot"
            blocks = soup.find_all(['p', 'div', 'section'], class_=re.compile(r'review|typography'))
        elif 'yelp' in domain:
            source = "Yelp"
            blocks = soup.find_all(['p', 'span'], class_=re.compile(r'comment|review'))
        elif 'g2' in domain or 'capterra' in domain:
            source = "G2/Capterra"
            blocks = soup.find_all(['div', 'p'], class_=re.compile(r'review|comment|content'))
        else:
            source = domain.split('.')[0].capitalize()
            blocks = soup.find_all(['p', 'li', 'blockquote'])

        min_len = 50 if 'trustpilot' in domain or 'yelp' in domain else 60
        for b in blocks:
            text = b.get_text(separator=' ', strip=True)
            if len(text) > min_len:
                feedbacks.append(text[:500])

        seen, unique = set(), []
        for f in feedbacks:
            if f not in seen and len(f.split()) > 8:
                seen.add(f)
                unique.append(f)
        feedbacks = unique[:100]

        if not feedbacks:
            error = "⚠️ No readable feedback found. The site may require login or use JavaScript rendering."

    except requests.exceptions.Timeout:
        error = "⏱️ Request timed out."
    except requests.exceptions.HTTPError as e:
        error = f"🚫 HTTP Error: {e}"
    except Exception as e:
        error = f"❌ Scraping failed: {str(e)}"

    return feedbacks, source, error

# ── Helper: Topic Modeling ─────────────────────────────────────────────────────
def run_topic_model(texts, n_topics=5):
    if len(texts) < 2:
        return [0] * len(texts)
    n_topics = min(n_topics, len(texts))
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500, min_df=1)
        X = vectorizer.fit_transform(texts)
        W = NMF(n_components=n_topics, random_state=42, max_iter=300).fit_transform(X)
        return W.argmax(axis=1).tolist()
    except Exception:
        return [0] * len(texts)

# ── Helper: Run AI suggestions with progress ───────────────────────────────────
def run_suggestions_with_progress(fb_df):
    all_sugs = []
    if openai.api_key:
        batch_size = 10
        prog = st.progress(0)
        stat = st.empty()
        total_b = max(1, len(fb_df) // batch_size + (1 if len(fb_df) % batch_size else 0))
        for i in range(0, len(fb_df), batch_size):
            batch = fb_df['Feedback'].iloc[i:i+batch_size].tolist()
            bn = i // batch_size + 1
            stat.markdown(
                f'<p class="pulse" style="color:#f9a825;font-family:Space Mono,monospace;font-size:0.8rem;">'
                f'🤖 AI analyzing batch {bn}/{total_b}...</p>',
                unsafe_allow_html=True
            )
            all_sugs.extend(get_suggestions_openai(batch))
            prog.progress(min(1.0, (i + batch_size) / len(fb_df)))
            time.sleep(0.3)
        stat.empty()
        prog.empty()
    else:
        all_sugs = [""] * len(fb_df)
    return all_sugs

# ── Helper: Build Results DF ───────────────────────────────────────────────────
def build_results(feedbacks, source_label):
    df = pd.DataFrame({'Feedback': feedbacks, 'Source': source_label})
    df['CleanedFeedback'] = df['Feedback'].apply(preprocess_text)
    df = df[df['CleanedFeedback'].str.strip() != ''].reset_index(drop=True)
    if df.empty:
        return df
    df['TopicID'] = run_topic_model(df['CleanedFeedback'].tolist())
    df['Topic'] = df['TopicID'].map(TOPIC_LABELS)
    all_sugs = run_suggestions_with_progress(df)
    df['Suggestion'] = all_sugs[:len(df)]
    df['Suggestion'] = df.apply(
        lambda r: r['Suggestion'] if isinstance(r['Suggestion'], str) and r['Suggestion'].strip()
        else FALLBACK_SUGGESTIONS.get(r['TopicID'], "Review feedback for patterns."),
        axis=1
    )
    return df[['Source', 'Feedback', 'Topic', 'Suggestion']]

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.65rem;letter-spacing:3px;text-transform:uppercase;color:#f9a825;margin-bottom:0.5rem;">⚙ Configuration</div>', unsafe_allow_html=True)

    if openai.api_key:
        st.success("✅ OpenAI API key loaded from .env")
    else:
        st.error("❌ OPENAI_API_KEY not found in .env — fallback suggestions will be used")

    st.divider()

    if SELENIUM_AVAILABLE:
        st.success("✅ Selenium ready — Quora scraping enabled")
    else:
        st.warning("⚠️ Selenium not installed — Quora won't work\n\n`pip install selenium webdriver-manager`")

    st.divider()
    st.markdown("**📌 Supported Sources**")
    for p in ["📄 CSV File Upload", "🔗 Quora", "🤖 Reddit", "⭐ Trustpilot", "🍕 Yelp", "💼 G2 / Capterra", "🌐 Any Public URL"]:
        st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:0.72rem;color:#aaa;margin:4px 0;">• {p}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.6rem;color:#555;text-align:center;">FeedbackIQ v2.0 · Powered by GPT-4</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════
st.markdown('<div class="hero-title">ZEUS FEEDBACK ANALIZER</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Customer Intelligence · Topic Modeling · AI Insights</div>', unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

tab1, tab2, tab3 = st.tabs(["📄 CSV Upload", "🔗 URL Scraper", "📊 Results"])

# ══════════════════════════════════════════════════════
# TAB 1 — CSV Upload
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="card"><div class="card-title">📤 Upload Feedback CSV</div>', unsafe_allow_html=True)
    st.markdown("Upload a **CSV file** containing customer feedback. The app will auto-detect the feedback column.")

    uploaded = st.file_uploader("Drop your CSV here", type=['csv'], label_visibility="collapsed")
    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            df_raw.columns = df_raw.columns.str.strip()
            st.success(f"✅ Loaded **{len(df_raw):,}** rows · Columns: `{'`, `'.join(df_raw.columns.tolist())}`")

            text_cols = [c for c in df_raw.columns if df_raw[c].dtype == object]
            default_col = next((c for c in text_cols if any(k in c.lower() for k in ['feedback', 'comment', 'review', 'text', 'response'])), text_cols[0] if text_cols else None)

            if not text_cols:
                st.error("❌ No text columns found in CSV.")
            else:
                selected_col = st.selectbox("Select the feedback column", text_cols, index=text_cols.index(default_col) if default_col else 0)
                st.dataframe(df_raw[[selected_col]].head(5), use_container_width=True)

                if st.button("🚀 Analyze CSV Feedback", key="csv_btn"):
                    feedbacks = df_raw[selected_col].dropna().astype(str).tolist()
                    with st.spinner("Processing..."):
                        results = build_results(feedbacks, "CSV Upload")
                    if results.empty:
                        st.error("❌ No processable feedback found after cleaning.")
                    else:
                        st.session_state.results_df = results
                        st.session_state.analyzed = True
                        st.success(f"✅ Analysis complete! **{len(results):,}** entries processed.")
                        st.info("👉 Switch to the **📊 Results** tab to explore insights.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📖 CSV Format Guide"):
        st.markdown("""
**Column names auto-detected:** `feedback`, `comment`, `review`, `text`, `response`

**Example:**
```
FeedbackCol,Rating
"Delivery was very slow and packaging damaged",2
"Great product quality! Will buy again.",5
```
        """)

# ══════════════════════════════════════════════════════
# TAB 2 — URL Scraper
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="card"><div class="card-title">🌐 Scrape Feedback from URL</div>', unsafe_allow_html=True)
    st.markdown("Paste **any public URL** — Trustpilot, Yelp, Reddit, G2, or any review/forum page.")

    url_input = st.text_input("Paste URL here", placeholder="https://www.trustpilot.com/...", label_visibility="collapsed")
    _, col2 = st.columns([3, 1])
    with col2:
        multi_mode = st.checkbox("Multi-URL mode")

    if multi_mode:
        multi_urls = st.text_area("Enter multiple URLs (one per line)", placeholder="https://...\nhttps://...", height=120, label_visibility="collapsed")

    if st.button("🔍 Scrape & Analyze", key="url_btn"):
        urls_to_scrape = []
        if multi_mode and multi_urls.strip():
            urls_to_scrape = [u.strip() for u in multi_urls.strip().split('\n') if u.strip().startswith('http')]
        elif url_input.strip():
            urls_to_scrape = [url_input.strip()]

        if not urls_to_scrape:
            st.warning("⚠️ Please enter at least one valid URL starting with http:// or https://")
        else:
            all_feedbacks, all_sources = [], []
            for url in urls_to_scrape:
                label = f"`{url[:60]}...`" if len(url) > 60 else f"`{url}`"
                with st.status(f"🔄 Scraping: {label}"):
                    feedbacks, source, error = scrape_url(url)
                    if error:
                        st.error(error)
                    else:
                        st.success(f"✅ Found **{len(feedbacks)}** entries from **{source}**")
                        all_feedbacks.extend(feedbacks)
                        all_sources.extend([source] * len(feedbacks))

            if all_feedbacks:
                st.info(f"📊 Total: **{len(all_feedbacks)}** entries scraped. Running analysis...")
                fb_df = pd.DataFrame({'Feedback': all_feedbacks, '_source': all_sources})
                fb_df['CleanedFeedback'] = fb_df['Feedback'].apply(preprocess_text)
                fb_df = fb_df[fb_df['CleanedFeedback'].str.strip() != ''].reset_index(drop=True)
                fb_df['TopicID'] = run_topic_model(fb_df['CleanedFeedback'].tolist())
                fb_df['Topic'] = fb_df['TopicID'].map(TOPIC_LABELS)
                all_sugs = run_suggestions_with_progress(fb_df)
                fb_df['Suggestion'] = all_sugs[:len(fb_df)]
                fb_df['Suggestion'] = fb_df.apply(
                    lambda r: r['Suggestion'] if isinstance(r['Suggestion'], str) and r['Suggestion'].strip()
                    else FALLBACK_SUGGESTIONS.get(r['TopicID'], "Review this feedback further."),
                    axis=1
                )
                results = fb_df.rename(columns={'_source': 'Source'})[['Source', 'Feedback', 'Topic', 'Suggestion']]
                st.session_state.results_df = results
                st.session_state.analyzed = True
                st.success(f"✅ Done! **{len(results):,}** entries analyzed.")
                st.info("👉 Switch to the **📊 Results** tab to explore insights.")
            else:
                st.error("No feedback could be extracted. Try a different URL or check if the site requires login.")

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("⚠️ Scraping Limitations"):
        st.markdown("""
- ✅ Trustpilot, G2, Yelp review pages
- ✅ Reddit threads (`old.reddit.com` works best)
- ✅ **Quora** — now supported via Selenium (requires Chrome installed)
- ❌ Facebook / Instagram — login-gated
- ❌ JavaScript-heavy SPAs (login required)

**Quora tip:** Make sure Chrome browser is installed on your machine. Selenium will auto-download the matching ChromeDriver.

**General tip:** If scraping fails, paste feedback into a CSV and use the CSV Upload tab.
        """)

# ══════════════════════════════════════════════════════
# TAB 3 — Results
# ══════════════════════════════════════════════════════
with tab3:
    if not st.session_state.analyzed or st.session_state.results_df is None:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;">
            <div style="font-size:4rem;margin-bottom:1rem;">📭</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.9rem;color:#666;letter-spacing:2px;text-transform:uppercase;">No results yet</div>
            <div style="color:#555;margin-top:0.5rem;font-size:0.85rem;">Upload a CSV or scrape a URL first</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state.results_df
        total = len(df)
        topic_counts = df['Topic'].value_counts()

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Entries", f"{total:,}")
        c2.metric("Topics Found", df['Topic'].nunique())
        c3.metric("Data Sources", df['Source'].nunique())
        c4.metric("AI Suggestions", total)

        st.divider()

        # Topic distribution
        st.markdown("### 🗂️ Topic Distribution")
        cols = st.columns(len(topic_counts))
        for i, (topic, count) in enumerate(topic_counts.items()):
            pct = round(count / total * 100)
            with cols[i]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                            border-radius:12px;padding:1rem;text-align:center;">
                    <div style="font-size:1.6rem;font-weight:800;color:#f9a825;font-family:'Space Mono',monospace;">{count}</div>
                    <div style="font-size:0.68rem;color:#aaa;margin:4px 0;">{topic}</div>
                    <div style="font-size:0.75rem;color:#666;">{pct}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Filters
        st.markdown("### 🔍 Browse & Filter")
        cf1, cf2 = st.columns(2)
        with cf1:
            topic_filter = st.multiselect("Filter by Topic", df['Topic'].unique().tolist(), default=df['Topic'].unique().tolist())
        with cf2:
            source_filter = st.multiselect("Filter by Source", df['Source'].unique().tolist(), default=df['Source'].unique().tolist())

        filtered = df[df['Topic'].isin(topic_filter) & df['Source'].isin(source_filter)]
        st.markdown(f'<div style="font-family:Space Mono,monospace;font-size:0.75rem;color:#888;margin-bottom:1rem;">Showing {len(filtered):,} of {total:,} entries</div>', unsafe_allow_html=True)

        view_mode = st.radio("View as", ["📋 Table", "🃏 Cards"], horizontal=True)

        if view_mode == "📋 Table":
            st.dataframe(
                filtered[['Source', 'Feedback', 'Topic', 'Suggestion']],
                use_container_width=True,
                height=400,
                column_config={
                    "Feedback": st.column_config.TextColumn("Feedback", width="large"),
                    "Suggestion": st.column_config.TextColumn("AI Suggestion", width="large"),
                    "Topic": st.column_config.TextColumn("Topic", width="medium"),
                    "Source": st.column_config.TextColumn("Source", width="small"),
                }
            )
        else:
            for _, row in filtered.head(30).iterrows():
                tid = [k for k, v in TOPIC_LABELS.items() if v == row['Topic']]
                badge_cls = TOPIC_COLORS.get(tid[0] if tid else 0, "badge-0")
                st.markdown(f"""
                <div class="card" style="margin-bottom:1rem;">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.8rem;">
                        <span class="source-tag">{row['Source']}</span>
                        <span class="badge {badge_cls}">{row['Topic']}</span>
                    </div>
                    <div style="color:#ddd;font-size:0.9rem;line-height:1.6;margin-bottom:0.8rem;">
                        {row['Feedback'][:300]}{'...' if len(row['Feedback']) > 300 else ''}
                    </div>
                    <div style="border-top:1px solid rgba(255,255,255,0.06);padding-top:0.8rem;">
                        <span style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#f9a825;letter-spacing:2px;text-transform:uppercase;">💡 Suggestion</span>
                        <div style="color:#b0b0b0;font-size:0.85rem;margin-top:4px;">{row['Suggestion']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            if len(filtered) > 30:
                st.info(f"Showing first 30 cards. Switch to Table view to see all {len(filtered)} entries.")

        st.divider()

        # Export
        st.markdown("### 📥 Export Results")
        cd1, cd2 = st.columns(2)
        with cd1:
            st.download_button("⬇️ Download CSV", data=filtered.to_csv(index=False).encode('utf-8'), file_name="feedbackiq_results.csv", mime='text/csv', use_container_width=True)
        with cd2:
            lines = ["FeedbackIQ Analysis Summary", "=" * 40, ""]
            for topic, count in topic_counts.items():
                lines.append(f"{topic}: {count} entries ({round(count/total*100)}%)")
            lines += ["", "Top Suggestions by Topic", "-" * 30]
            for tid, label in TOPIC_LABELS.items():
                subset = df[df['Topic'] == label]
                if not subset.empty:
                    lines += [f"\n{label}", f"→ {subset.iloc[0]['Suggestion']}"]
            st.download_button("⬇️ Download Summary Report", data='\n'.join(lines).encode('utf-8'), file_name="feedbackiq_summary.txt", mime='text/plain', use_container_width=True)

        st.divider()
        if st.button("🔄 Clear & Start Over"):
            st.session_state.results_df = None
            st.session_state.analyzed = False
            st.rerun()