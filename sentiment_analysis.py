import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def sentiment_prediction(df):
    df["Polarity"] = df["Clean text"].apply(sentiment_polarity)
    df["Sentiment"] = df["Polarity"].apply(classify_sentiment)
    return df

def sentiment_polarity(text):
    analyzer = SentimentIntensityAnalyzer()
    polarity_score = analyzer.polarity_scores(text)["compound"]
    return polarity_score

def classify_sentiment(polarity):
    if polarity < 0:
        return -1
    elif polarity == 0:
        return 0
    else:
        return 1