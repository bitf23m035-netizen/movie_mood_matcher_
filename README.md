# Movie Mood Matcher

> Tell it your mood. Get a movie. That's it.

A machine learning web app that predicts your ideal movie mood category from 5 quick questions, then recommends the highest-rated films that match — including Hollywood, Bollywood, and Lollywood options.

## Live Demo
[Open App](https://moviemoodmatcher-tsibtjmbb3se8uqs4gmewz.streamlit.app/) · [Model on Hugging Face](https://huggingface.co/devnotebook/movie-mood-matcher/tree/main)

## How it works
1. You answer 5 questions about your mood, energy, company, preferred length, and industry
2. A trained Naive Bayes classifier predicts your mood category
3. App filters MovieLens movies by that category and shows top-rated picks

## Mood Categories
| Mood | Genres |
|------|--------|
| Happy | Comedy, Animation, Musical |
| Thrilled | Action, Adventure, Sci-Fi |
| Dreamy | Romance, Drama |
| Scared | Horror, Thriller, Mystery |
| Thoughtful | Documentary, History |
| Chill | Family, Fantasy |

## Model Comparison
| Algorithm | Accuracy |
|-----------|----------|
| Naive Bayes | 75.53% |
| Logistic Regression | 48.74% |

Naive Bayes outperformed Logistic Regression on this dataset because mood features are largely independent categorical variables — exactly where NB excels.

## Dataset
- MovieLens Small Dataset — 9,742 movies, 100,836 ratings (GroupLens)
- Custom Bollywood/Lollywood additions — 20 manually curated films

## Tech Stack
Python · scikit-learn · Streamlit · pandas · Hugging Face

## Run Locally
pip install -r requirements.txt
streamlit run app/app.py
