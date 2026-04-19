import streamlit as st # type: ignore

st.set_page_config(page_title='Movie Mood Matcher', page_icon='🎬', layout='centered')

import pickle
import pandas as pd # type: ignore
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_models():
    model = pickle.load(open(os.path.join(BASE, 'model', 'mood_classifier.pkl'), 'rb'))
    le    = pickle.load(open(os.path.join(BASE, 'model', 'label_encoder.pkl'),   'rb'))
    oe    = pickle.load(open(os.path.join(BASE, 'model', 'ordinal_encoder.pkl'), 'rb'))
    return model, le, oe

@st.cache_data
def load_movies():
    return pd.read_csv(os.path.join(BASE, 'data', 'mood_movies.csv'))

model, le, oe = load_models()
movies_df     = load_movies()

MOOD_COLORS = {
    'Happy':      '#1D9E75',
    'Thrilled':   '#378ADD',
    'Dreamy':     '#D4537E',
    'Scared':     '#E24B4A',
    'Thoughtful': '#7F77DD',
    'Chill':      '#BA7517',
}

MOOD_MESSAGES = {
    'Happy':      'You need something feel-good and fun!',
    'Thrilled':   'Buckle up — time for pure adrenaline!',
    'Dreamy':     'A cozy night in calls for something dreamy.',
    'Scared':     'Feeling brave? Let us make it worse.',
    'Thoughtful': 'Your brain wants something deep tonight.',
    'Chill':      'Low effort, high comfort. You deserve it.',
}

MOOD_EMOJIS = {
    'Happy': '😊', 'Thrilled': '⚡', 'Dreamy': '🌙',
    'Scared': '👻', 'Thoughtful': '🧠', 'Chill': '☕'
}

st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 0.2rem;
}
.sub-title {
    text-align: center;
    color: #888;
    font-size: 1rem;
    margin-bottom: 2rem;
}
.result-box {
    padding: 20px 24px;
    border-radius: 14px;
    border-left: 5px solid;
    margin-bottom: 20px;
}
.movie-card {
    padding: 12px 16px;
    border-radius: 10px;
    background: #f8f8f8;
    margin-bottom: 10px;
    border: 1px solid #eee;
}
.question-label {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Movie Mood Matcher</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Answer 5 questions — get your perfect movie tonight</div>', unsafe_allow_html=True)

st.markdown("### How are you feeling right now?")
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="question-label">Your current mood</div>', unsafe_allow_html=True)
    mood = st.select_slider('',
        options=['Sad', 'Chill', 'Neutral', 'Happy', 'Excited'],
        key='mood')

    st.markdown('<div class="question-label">Energy level right now</div>', unsafe_allow_html=True)
    energy = st.select_slider('',
        options=['Exhausted', 'Low', 'Medium', 'High', 'Hyper'],
        key='energy')

    st.markdown('<div class="question-label">Watching with?</div>', unsafe_allow_html=True)
    company = st.selectbox('', ['Solo', 'Partner', 'Friends', 'Family'], key='company')

with col2:
    st.markdown('<div class="question-label">How long a movie?</div>', unsafe_allow_html=True)
    length = st.selectbox('', ['Any', 'Short (<90 min)', 'Long (>120 min)'], key='length')

    st.markdown('<div class="question-label">Preferred industry?</div>', unsafe_allow_html=True)
    industry = st.selectbox('', ['Any', 'Hollywood', 'Bollywood', 'Lollywood'], key='industry')

st.markdown("---")

if st.button('Find my movie tonight', type='primary', use_container_width=True):

    inp = pd.DataFrame(
        [[mood, energy, company, length, 'Any']],
        columns=['mood_input', 'energy', 'company', 'length_pref', 'language_pref']
    )

    try:
        enc      = oe.transform(inp)
        pred     = model.predict(enc)[0]
        proba    = model.predict_proba(enc)[0]
        category = le.inverse_transform([pred])[0]
        confidence = round(max(proba) * 100)
    except Exception:
        inp['language_pref'] = 'English'
        enc      = oe.transform(inp)
        pred     = model.predict(enc)[0]
        proba    = model.predict_proba(enc)[0]
        category = le.inverse_transform([pred])[0]
        confidence = round(max(proba) * 100)

    color   = MOOD_COLORS.get(category, '#888')
    message = MOOD_MESSAGES.get(category, '')
    emoji   = MOOD_EMOJIS.get(category, '')

    st.markdown(f"""
    <div class="result-box" style="border-color:{color}; background:{color}11;">
        <div style="font-size:1.6rem; font-weight:700; color:{color}">{emoji} {category}</div>
        <div style="font-size:0.95rem; margin-top:6px; color:#555;">{message}</div>
        <div style="font-size:0.8rem; margin-top:4px; color:#aaa;">Confidence: {confidence}%</div>
    </div>
    """, unsafe_allow_html=True)

    recs = movies_df[movies_df['mood_category'] == category].copy()

    if industry == 'Bollywood':
        boll = recs[recs['industry'] == 'Bollywood'] if 'industry' in recs.columns else pd.DataFrame()
        recs = boll if len(boll) > 0 else recs
    elif industry == 'Lollywood':
        loll = recs[recs['industry'] == 'Lollywood'] if 'industry' in recs.columns else pd.DataFrame()
        recs = loll if len(loll) > 0 else recs

    recs = recs.sort_values('avg_rating', ascending=False).drop_duplicates('title').head(5)

    st.markdown("### Tonight's picks for you")
    for _, row in recs.iterrows():
        ind_badge = f" · {row['industry']}" if 'industry' in row and pd.notna(row.get('industry')) else ''
        st.markdown(f"""
        <div class="movie-card">
            <div style="font-weight:600; font-size:1rem;">{row['title']}</div>
            <div style="font-size:0.85rem; color:#888; margin-top:3px;">{row['genres']}{ind_badge} &nbsp;⭐ {row['avg_rating']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)