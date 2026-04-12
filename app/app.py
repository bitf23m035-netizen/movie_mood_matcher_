import streamlit as st

st.set_page_config(page_title='Movie Mood Matcher', page_icon='🎬', layout='centered')

import pickle
import pandas as pd
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
    'Thrilled':   'Buckle up — time for pure adrenaline.',
    'Dreamy':     'A cozy night in calls for something dreamy.',
    'Scared':     'Feeling brave? Let us make it worse.',
    'Thoughtful': 'Your brain wants something deep tonight.',
    'Chill':      'Low effort, high comfort. You deserve it.',
}

st.title('Movie Mood Matcher')
st.write('Answer 5 questions. Get the perfect movie for tonight.')
st.divider()

col1, col2 = st.columns(2)
with col1:
    mood    = st.select_slider('How are you feeling?',
                options=['Sad','Chill','Neutral','Happy','Excited'])
    energy  = st.select_slider('Energy level?',
                options=['Exhausted','Low','Medium','High','Hyper'])
    company = st.selectbox('Watching with?',
                ['Solo','Partner','Friends','Family'])
with col2:
    length   = st.selectbox('Movie length?',
                 ['Short (<90 min)','Any','Long (>120 min)'])
    language = st.selectbox('Language?',
                 ['English','Any','Non-English'])

st.divider()

if st.button('Find my movie', type='primary', use_container_width=True):
    inp  = pd.DataFrame([[mood, energy, company, length, language]],
             columns=['mood_input','energy','company','length_pref','language_pref'])
    enc  = oe.transform(inp)
    pred = model.predict(enc)[0]
    proba    = model.predict_proba(enc)[0]
    category = le.inverse_transform([pred])[0]
    confidence = round(max(proba) * 100)

    color   = MOOD_COLORS.get(category, '#888')
    message = MOOD_MESSAGES.get(category, '')

    st.markdown(f"""
    <div style="padding:16px 20px; border-radius:12px;
         border-left:4px solid {color}; background:#f9f9f9;
         margin-bottom:16px;">
      <div style="font-size:22px; font-weight:600; color:{color}">{category}</div>
      <div style="font-size:14px; margin-top:4px; color:#555;">
          {message} · {confidence}% confidence
      </div>
    </div>""", unsafe_allow_html=True)

    recs = movies_df[movies_df['mood_category'] == category]\
               .sort_values('avg_rating', ascending=False)\
               .drop_duplicates('title')\
               .head(5)

    st.subheader('Tonight picks for you')
    for _, row in recs.iterrows():
        st.markdown(f"**{row['title']}**  \n"
                    f"{row['genres']}  ·  ⭐ {row['avg_rating']:.1f}")
        st.divider()