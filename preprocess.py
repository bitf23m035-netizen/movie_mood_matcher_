import pandas as pd
import random

random.seed(42)

movies  = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
avg_ratings.columns = ['movieId', 'avg_rating']
avg_ratings['avg_rating'] = avg_ratings['avg_rating'].round(2)

df = movies.merge(avg_ratings, on='movieId')

MOOD_MAP = {
    'Happy':      ['Comedy','Animation','Musical'],
    'Thrilled':   ['Action','Adventure','Sci-Fi'],
    'Dreamy':   ['Romance','Drama'],
    'Scared':     ['Horror','Thriller','Mystery'],
    'Thoughtful': ['Documentary','History'],
    'Chill':      ['Family','Fantasy'],
}

def assign_mood(genres_str):
    genres = genres_str.split('|')
    for mood, genre_list in MOOD_MAP.items():
        for g in genre_list:
            if g in genres:
                return mood
    return 'Happy'

df['mood_category'] = df['genres'].apply(assign_mood)

ENERGY  = ['Exhausted','Low','Medium','High','Hyper']
COMPANY = ['Solo','Partner','Friends','Family']
LENGTH  = ['Short (<90 min)','Any','Long (>120 min)']
LANG    = ['English','Any','Non-English']

MOOD_TO_INPUT = {
    'Happy':      ['Happy','Neutral'],
    'Thrilled':   ['Excited','Happy'],
    'Dreamy':   ['Chill','Sad'],
    'Scared':     ['Excited','Neutral'],
    'Thoughtful': ['Sad','Neutral','Chill'],
    'Chill':      ['Chill','Sad'],
}
MOOD_TO_ENERGY = {
    'Happy':      ['Medium','High'],
    'Thrilled':   ['High','Hyper'],
    'Dreamy':   ['Low','Medium'],
    'Scared':     ['High','Hyper'],
    'Thoughtful': ['Exhausted','Low'],
    'Chill':      ['Exhausted','Low','Medium'],
}

df['mood_input']    = df['mood_category'].apply(lambda m: random.choice(MOOD_TO_INPUT[m]))
df['energy']        = df['mood_category'].apply(lambda m: random.choice(MOOD_TO_ENERGY[m]))
df['company']       = [random.choice(COMPANY) for _ in range(len(df))]
df['length_pref']   = [random.choice(LENGTH)  for _ in range(len(df))]
df['language_pref'] = [random.choice(LANG)    for _ in range(len(df))]

df.to_csv('data/mood_movies.csv', index=False)

print(f"Done! {len(df)} movies saved.")
print(df['mood_category'].value_counts())