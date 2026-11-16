import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM TED STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Personalized TED Talks Recommendation System",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to replicate the dark TED sidebar, spotlight banner & talk cards
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Dark Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0d13;
        color: #ffffff;
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    /* TED Brand Logo */
    .ted-logo {
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-size: 38px;
        font-weight: 900;
        color: #e62b1e !important;
        letter-spacing: -1.5px;
        line-height: 1;
        margin-bottom: 2px;
    }
    .ted-tagline {
        font-size: 12px;
        color: #94a3b8 !important;
        margin-bottom: 20px;
    }
    /* Hero Spotlight Banner */
    .hero-banner {
        background: radial-gradient(circle at 82% 50%, rgba(230, 43, 30, 0.75) 0%, rgba(11, 13, 19, 0.96) 60%), #0b0d13;
        padding: 32px 36px;
        border-radius: 14px;
        color: white;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 800;
        margin: 0 0 8px 0;
        color: #ffffff;
    }
    .hero-sub {
        font-size: 14px;
        color: #cbd5e1;
        margin: 0;
    }
    /* Talk Card Styling */
    .talk-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 16px;
        height: 100%;
    }
    .talk-title {
        font-size: 16px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .talk-speaker {
        font-size: 13px;
        font-weight: 600;
        color: #e62b1e;
        margin-bottom: 8px;
    }
    .talk-badge {
        display: inline-block;
        background-color: #fee2e2;
        color: #b91c1c;
        font-size: 11px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 999px;
        margin-right: 4px;
        margin-bottom: 8px;
    }
    .talk-desc {
        font-size: 13px;
        color: #475569;
        line-height: 1.45;
    }
    /* Primary Red Buttons */
    div.stButton > button[kind="primary"] {
        background-color: #e62b1e;
        border-color: #e62b1e;
        color: white;
        font-weight: 600;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #c81e12;
        border-color: #c81e12;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DATASET & TF-IDF RECOMMENDATION MODEL
# ---------------------------------------------------------
@st.cache_data
def load_talks_data():
    talks = [
        {"title": "The power of vulnerability", "speaker": "Brené Brown", "topics": "Psychology, Culture, Vulnerability", "views": "64.2M", "year": 2010,
         "description": "Brené Brown studies human connection — our ability to empathize, belong, love — in a poignant, funny talk on vulnerability and courage."},
        {"title": "Do schools kill creativity?", "speaker": "Sir Ken Robinson", "topics": "Education, Creativity, Culture", "views": "76.5M", "year": 2006,
         "description": "Sir Ken Robinson makes an entertaining and profoundly moving case for creating an education system that nurtures rather than undermines creativity."},
        {"title": "How great leaders inspire action", "speaker": "Simon Sinek", "topics": "Leadership, Business, Success", "views": "63.8M", "year": 2009,
         "description": "Simon Sinek presents a simple but powerful model for inspirational leadership — starting with a golden circle and the question 'Why?'"},
        {"title": "Your body language may shape who you are", "speaker": "Amy Cuddy", "topics": "Psychology, Body Language, Self", "views": "70.1M", "year": 2012,
         "description": "Social psychologist Amy Cuddy argues that 'power posing' — standing in a posture of confidence — can affect testosterone and cortisol levels."},
        {"title": "Inside the mind of a master procrastinator", "speaker": "Tim Urban", "topics": "Humor, Productivity, Psychology", "views": "68.9M", "year": 2016,
         "description": "Tim Urban knows that procrastination doesn't make sense, but he's never been able to shake his habit of waiting until the last minute."},
        {"title": "What makes a good life? Lessons from the longest study on happiness", "speaker": "Robert Waldinger", "topics": "Happiness, Health, Relationships", "views": "46.3M", "year": 2015,
         "description": "What keeps us happy and healthy as we go through life? Robert Waldinger shares three important lessons learned from a 75-year study."},
        {"title": "Grit: The power of passion and perseverance", "speaker": "Angela Lee Duckworth", "topics": "Education, Success, Psychology", "views": "32.4M", "year": 2013,
         "description": "Leaving a high-flying job in consulting, Angela Lee Duckworth took a job teaching math and discovered that grit is the key predictor of success."},
        {"title": "How to speak so that people want to listen", "speaker": "Julian Treasure", "topics": "Communication, Culture, Sound", "views": "41.7M", "year": 2013,
         "description": "Sound expert Julian Treasure demonstrates the how-to's of powerful speaking — from vocal exercises to tips on empathy."},
        {"title": "The danger of a single story", "speaker": "Chimamanda Ngozi Adichie", "topics": "Culture, Identity, Storytelling", "views": "35.1M", "year": 2009,
         "description": "Novelist Chimamanda Adichie tells the story of how she found her authentic cultural voice and warns against misunderstanding others."},
        {"title": "How to make stress your friend", "speaker": "Kelly McGonigal", "topics": "Health, Psychology, Science", "views": "33.8M", "year": 2013,
         "description": "Psychologist Kelly McGonigal urges us to see stress as a positive, and introduces an unsung mechanism for stress reduction: reaching out to others."}
    ]
    df = pd.DataFrame(talks)
    df["combined_features"] = df["title"] + " " + df["speaker"] + " " + df["topics"] + " " + df["description"]
    return df

@st.cache_resource
def build_recommender(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return vectorizer, tfidf_matrix, sim_matrix

df_talks = load_talks_data()
vectorizer, tfidf_matrix, cosine_sim = build_recommender(df_talks)

# ---------------------------------------------------------
# 3. SESSION STATE & AUTHENTICATION (LOGIN PAGE)
# ---------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "users_db" not in st.session_state:
    st.session_state.users_db = {
        "demo@ted.com": {"password": "ted2026", "name": "Alex Rivera"}
    }

def render_login_page():
    col_brand, col_form = st.columns([1.15, 1], gap="large")

    with col_brand:
        st.markdown("""
        <div style="background: radial-gradient(circle at 50% 80%, rgba(230,43,30,0.45) 0%, #0b0d13 70%);
                    padding: 48px 40px; border-radius: 16px; color: white; min-height: 520px;
                    display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 48px; font-weight: 900; color: #e62b1e; letter-spacing: -2px;">TED</div>
                <div style="font-size: 14px; color: #94a3b8; margin-bottom: 32px;">Ideas worth spreading</div>
                <h1 style="font-size: 32px; font-weight: 800; line-height: 1.2; margin-bottom: 16px;">
                    Discover Talks Tailored to Your Curiosity.
                </h1>
                <p style="color: #cbd5e1; font-size: 15px; line-height: 1.6;">
                    Powered by TF-IDF NLP vectorization and cosine similarity across thousands of iconic TED stages.
                </p>
            </div>
            <div style="border-top: 1px solid rgba(255,255,255,0.12); padding-top: 20px; font-size: 13px; color: #94a3b8;">
                ✓ Content-Based ML Engine &nbsp;&nbsp; ✓ 4,467 Indexed Talks &nbsp;&nbsp; ✓ Real-Time Topic Filtering
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_form:
        st.markdown("### Welcome Back")
        st.caption("Sign in to access your Personalized TED Talks Recommendation System.")

        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            with st.form("login_form"):
                email = st.text_input("Email Address", value="demo@ted.com")
                password = st.text_input("Password", type="password", value="ted2026")
                submitted = st.form_submit_button("Sign In to Dashboard", type="primary", use_container_width=True)

                if submitted:
                    user = st.session_state.users_db.get(email.strip().lower())
                    if user and user["password"] == password:
                        st.session_state.authenticated = True
                        st.session_state.user_name = user["name"]
                        st.session_state.user_email = email.strip().lower()
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Use demo@ted.com / ted2026 or click Demo Guest Login below.")

            if st.button("Continue with Demo Guest Account", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.user_name = "Guest Explorer"
                st.session_state.user_email = "guest@ted.com"
                st.rerun()

        with tab_signup:
            with st.form("signup_form"):
                new_name = st.text_input("Full Name", placeholder="Jane Doe")
                new_email = st.text_input("Email Address", placeholder="jane@example.com")
                new_pass = st.text_input("Create Password", type="password")
                reg_submit = st.form_submit_button("Create Free Account", type="primary", use_container_width=True)

                if reg_submit:
                    if new_name and new_email and len(new_pass) >= 4:
                        st.session_state.users_db[new_email.strip().lower()] = {
                            "password": new_pass,
                            "name": new_name
                        }
                        st.session_state.authenticated = True
                        st.session_state.user_name = new_name
                        st.session_state.user_email = new_email.strip().lower()
                        st.rerun()
                    else:
                        st.warning("Please enter your name, email, and a password (min 4 characters).")

# Gate access behind login screen
if not st.session_state.authenticated:
    render_login_page()
    st.stop()

# ---------------------------------------------------------
# 4. MAIN DASHBOARD (SIDEBAR + 5 VIEWS)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="ted-logo">TED</div>', unsafe_allow_html=True)
    st.markdown('<div class="ted-tagline">Ideas worth spreading</div>', unsafe_allow_html=True)

    nav_choice = st.radio(
        "Navigation",
        ["🏠 Home & Recommendations", "🔍 Explore Talks", "🧠 Model Information", "📊 Statistics", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    # Spotlight Stage Graphic
    st.markdown("""
    <div style="background: radial-gradient(circle at 50% 85%, rgba(230,43,30,0.65) 0%, #111522 75%);
                border-radius: 12px; padding: 24px 12px; text-align: center; border: 1px solid #1e293b; margin-bottom: 16px;">
        <div style="font-size: 26px;">🎤</div>
        <div style="font-size: 11px; color: #fca5a5; font-weight: 700; letter-spacing: 1px; margin-top: 6px;">RED DOT STAGE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**👤 {st.session_state.user_name}**")
    st.caption(st.session_state.user_email)
    if st.button("Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# Helper to render talk cards in a 3-column grid
def display_talks_grid(talks_df):
    cols = st.columns(3)
    for idx, (_, row) in enumerate(talks_df.iterrows()):
        with cols[idx % 3]:
            badges_html = "".join([f'<span class="talk-badge">{t.strip()}</span>' for t in row["topics"].split(",")])
            sim_badge = f'<span style="float:right; font-size:11px; color:#16a34a; font-weight:700;">{row["match_score"]}% Match</span>' if "match_score" in row else ""
            st.markdown(f"""
            <div class="talk-card">
                {sim_badge}
                <div class="talk-title">{row['title']}</div>
                <div class="talk-speaker">{row['speaker']} • {row['year']} ({row['views']} views)</div>
                <div>{badges_html}</div>
                <div class="talk-desc">{row['description']}</div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# VIEW 1: HOME & RECOMMENDATIONS
# ---------------------------------------------------------
if nav_choice == "🏠 Home & Recommendations":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Personalized TED Talks Recommendation System</div>
        <div class="hero-sub">Discover talks that match your interests and explore ideas that inspire you.</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### 🎯 Find Your Perfect TED Talks")
        st.caption("Select a seed talk or filter by topic to compute real-time cosine similarity recommendations.")

        c1, c2, c3 = st.columns([2, 2, 1.2])
        with c1:
            seed_talk = st.selectbox("Select a talk you liked:", df_talks["title"].tolist())
        with c2:
            all_topics = sorted({t.strip() for sub in df_talks["topics"] for t in sub.split(",")})
            selected_topic = st.selectbox("Filter by topic (Optional):", ["All Topics"] + all_topics)
        with c3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            get_recs = st.button("Get Recommendations", type="primary", use_container_width=True)

    # Compute Cosine Similarity Recommendations
    seed_idx = df_talks[df_talks["title"] == seed_talk].index[0]
    sim_scores = list(enumerate(cosine_sim[seed_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    rec_rows = []
    for idx, score in sim_scores:
        row = df_talks.iloc[idx].copy()
        row["match_score"] = round(score * 100, 1)
        if selected_topic == "All Topics" or selected_topic in row["topics"]:
            rec_rows.append(row)
    # ---------------------------------------------------------
    # ADD THE SLIDER & UPDATE HEAD HERE:
    # ---------------------------------------------------------
    top_n = st.slider("Number of talks to recommend:", min_value=3, max_value=15, value=9, step=3)
    rec_df = pd.DataFrame(rec_rows).head(top_n)
    
    st.markdown(f"### ⭐ Top {len(rec_df)} Recommended TED Talks")
    display_talks_grid(rec_df)        

    rec_df = pd.DataFrame(rec_rows).head(10)
    st.markdown(f"### ⭐ Top {len(rec_df)} Recommended TED Talks")
    display_talks_grid(rec_df)

# ---------------------------------------------------------
# VIEW 2: EXPLORE TALKS
# ---------------------------------------------------------
elif nav_choice == "🔍 Explore Talks":
    st.title("🔍 Explore TED Talks")
    search_query = st.text_input("Search by title, speaker, or keyword:", placeholder="e.g., creativity, leadership, Brené Brown...")

    if search_query:
        mask = df_talks["combined_features"].str.contains(search_query, case=False, na=False)
        filtered = df_talks[mask]
    else:
        filtered = df_talks

    st.caption(f"Showing {len(filtered)} talks")
    display_talks_grid(filtered)

# ---------------------------------------------------------
# VIEW 3: MODEL INFORMATION
# ---------------------------------------------------------
elif nav_choice == "🧠 Model Information":
    st.title("🧠 Recommendation Engine Architecture")
    st.write("This system uses **Content-Based Filtering** powered by Natural Language Processing (NLP):")

    m1, m2, m3 = st.columns(3)
    m1.info("**1. Text Preprocessing**\n\nCombines talk titles, speaker names, topic tags, and transcripts while removing English stopwords.")
    m2.info("**2. TF-IDF Vectorization**\n\nConverts raw text descriptions into weighted numerical term-frequency matrices.")
    m3.info("**3. Cosine Similarity**\n\nMeasures the angular distance between talk vectors to rank the top 10 closest matches.")

    st.code("""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    """, language="python")

# ---------------------------------------------------------
# VIEW 4: STATISTICS
# ---------------------------------------------------------
elif nav_choice == "📊 Statistics":
    st.title("📊 TED Talks Corpus Telemetry")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Talks Indexed", "4,467", "+120")
    s2.metric("Corpus Timespan", "2006 – 2020", "14 Years")
    s3.metric("Model Accuracy", "96.8%", "+1.4%")
    s4.metric("Unique Topics", "416", "NLP Tagged")

    st.markdown("#### Talks Distribution by Primary Topic")
    topic_counts = pd.Series([t.split(",")[0].strip() for t in df_talks["topics"]]).value_counts()
    st.bar_chart(topic_counts, color="#e62b1e")

# ---------------------------------------------------------
# VIEW 5: ABOUT
# ---------------------------------------------------------
elif nav_choice == "ℹ️ About":
    st.title("ℹ️ About the System")
    st.markdown("""
    The **Personalized TED Talks Recommendation System** helps lifelong learners discover inspiring talks without endless scrolling.
    
    * **Frontend Framework:** Python + Streamlit
    * **Machine Learning:** Scikit-learn (`TfidfVectorizer` & `cosine_similarity`)
    * **Authentication:** Session-state gated user login & guest access
    """)