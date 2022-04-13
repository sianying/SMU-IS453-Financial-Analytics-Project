from sklearn.feature_extraction.text import CountVectorizer
import nltk
# nltk.download('punkt')
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import re

from flask import Flask
from flask_socketio import SocketIO, emit
import os
import pandas as pd
import datetime as dt
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from web_scraper import etf_holdings_weights, twitter_scraper
from pipeline import *
from sentiment_analysis import sentiment_prediction

app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")


df = {}
nlp_lemmatize = spacy.load('en_core_web_sm')
nlp_stopwords = English()
tokens = []

@socketio.on('get-prediction')
def get_prediction(client_data):

    # print(client_data)
    stocks = client_data['stocks']
    combi = client_data['combi']

    print(stocks)
    print(combi)

    tickers = [stock.split(" ")[0] for stock in stocks]

    df = get_tickers_data(tickers)
    print(df)
    result = train_xgboost(df, combi)

    return {
        "code": "received!",
        "data": result
    }

@socketio.on('sentiment-wordcloud')
def get_sentiment_wordcloud(data):
    # print("df3", type(df))

    stocks = [stock.split(" ")[0] for stock in data['stocks']]
    combi = data['combi']

    print("stock", stocks)
    print("combi", combi)

    try:
        #read csv
        vgt_df = pd.read_csv(f'./assets/{stocks[0]}_sentiment.csv').drop(columns=["Unnamed: 0"])
        xle_df = pd.read_csv(f'./assets/{stocks[1]}_sentiment.csv').drop(columns=["Unnamed: 0"])
        xlv_df = pd.read_csv(f'./assets/{stocks[2]}_sentiment.csv').drop(columns=["Unnamed: 0"])

        vgt_polarity = vgt_df.groupby('Date')['Polarity Vader'].mean().to_frame()
        xle_polarity = xle_df.groupby('Date')['Polarity Vader'].mean().to_frame()
        xlv_polarity = xlv_df.groupby('Date')['Polarity Vader'].mean().to_frame()

        vgt_polarity = vgt_polarity * combi[0]
        xle_polarity = xle_polarity * combi[1]
        xlv_polarity = xlv_polarity * combi[2]

        vgt_polarity = vgt_polarity.add(xle_polarity)
        vgt_polarity = vgt_polarity.add(xlv_polarity)

        vgt_polarity = vgt_polarity[-7:]

        dates = list(vgt_polarity.index)
        polarity = list(vgt_polarity['Polarity Vader'])

        #wordcloud
        vgt_1_day = vgt_df[vgt_df.Date == '2022-03-24']['Clean text']
        xle_1_day = xle_df[xle_df.Date == '2022-03-24']['Clean text']
        xlv_1_day = xlv_df[xlv_df.Date == '2022-03-24']['Clean text']

        tweets = pd.concat([vgt_1_day, xle_1_day,xlv_1_day]).to_frame()

        tweets["Clean text"] = tweets["Clean text"].apply(clean_text_extended)
        tweets["Clean text"] = tweets["Clean text"].apply(lemmatize)
        tweets["Clean text"] = tweets["Clean text"].apply(remove_stopwords)
        tweets['Clean text'].apply(lambda x: tokenize(x, tokens))

        fdist = nltk.FreqDist(tokens)
        word_cloud_data = pd.DataFrame(data=fdist.most_common(100), columns=['x', 'value']).to_dict('records')
        print(word_cloud_data)

        data = {
            "dates": dates, 
            "polarity": polarity,
            "word_cloud_data": word_cloud_data
        }
        return data

    except Exception as e:
        print("error!! ", e)

    # etf_holdings_dict = etf_holdings_weights(query, 2)
    # query_list = list(etf_holdings_dict.keys())
    # #add ETF to holdings list
    # query_list.append(query)

    # emit('sentiment-notif', 'Scraping the web...')

    # tweets = twitter_scraper(query_list, 1)
    # print(len(tweets))

    # emit('sentiment-notif', 'Cleaning tweets...')
    # cleaned_tweets = text_preprocessing(tweets)

    # emit('sentiment-notif', 'Predicting sentiment polarity...')
    # tweets_sentiment = sentiment_prediction(cleaned_tweets)
    # print(tweets_sentiment[:3])

def get_tickers_data(tickers, time_period=365):
    now = dt.date.today()
    start = now - dt.timedelta(days=365)
    end = now
    tickers = ['VGT', 'XLE', 'XLV']
    try:
        df = yf.download(tickers, start, end)
        return df

        # df[df.index.duplicated(keep='first')]
        # close_price = list(df['Close'][tickers[0]])
        # dates = list(df.index.strftime("%Y-%m-%d"))
        # data = {
        #     "dates": dates, 
        #     "close_price": close_price
        # 

    except:
        print("error! failed to get Yahoo data")
        return "Failed! :("

def clean_text_extended(text):
    # Remove punctuations
    punctuations = "!?.,`"
    result = []
    for char in punctuations:
        text = text.replace(char, " ")
    # Replace ’ with '
    text = text.replace("’", "'")
    #replace multi spaces with 1 space
    text = re.sub('\s+'," ", text)
    # Remove non-aplha
    result = []
    for word in text.split(" "):
        is_numeric = False
        for char in word:
            if char.isnumeric():
                is_numeric = True
        if not is_numeric:
            result.append(word)
    text = " ".join(result)
    return text


def lemmatize(text):
    corpus = nlp_lemmatize(text)
    token_list = []
    for token in corpus:
        token_list.append(token)
    text = " ".join([token.lemma_ for token in token_list])
    text = text.replace("-PRON-", "")
    text = re.sub('\s+'," ", text)
    text = text.strip()
    return text

def remove_stopwords(text):
    corpus = nlp_stopwords(text)
    token_list = []
    for token in corpus:
        token_list.append(token.text)

    filtered_sentence = [] 
    for word in token_list:
        lexeme = nlp_stopwords.vocab[word]
        if lexeme.is_stop == False:
            filtered_sentence.append(word)
    text = " ".join(filtered_sentence)
    text = text.strip()
    return text

def tokenize(text, tokens):
    token = nltk.word_tokenize(text)
    for word in token:
        if word.isalpha():
            tokens.append(word)

def train_xgboost(df, combi):
    # print("df", df)
    new_data = df.drop(columns=["Close", "High", "Low", "Open", "Volume"])
    new_data = new_data.sort_index(ascending=True, axis=0)
    # print("new_data", new_data)
    # print("combi", combi)
    new_data = new_data * combi
    combined = new_data.sum(axis=1).to_frame()
    combined.rename(columns={0 :'Adj Close'}, inplace=True )
    combined['PCT_CHNG'] = (combined['Adj Close'].pct_change()+ 1).cumprod() - 1
    combined = combined.drop(columns=['Adj Close'])


    data = combined.sort_index(ascending=True, axis=0).dropna()
    data

    X_train, X_test, y_train, y_test = train_test_split(data['PCT_CHNG'], data['PCT_CHNG'], test_size=0.2, random_state=0, shuffle=False)

    #scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_train_x = scaler.fit_transform(X_train.values.reshape(-1, 1))
    scaled_data_test_x = scaler.transform(X_test.values.reshape(-1, 1))

    #scaler for inverse
    y_trained_reshaped = y_train.values.reshape(-1, 1)
    inverse_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data_train_y = inverse_scaler.fit_transform(y_trained_reshaped) #close price at position 0

    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.3, max_depth = 5, alpha = 10, n_estimators = 100)
    xg_reg.fit(scaled_data_train_x, scaled_data_train_y)
    predicted = xg_reg.predict(scaled_data_test_x)
    predicted = predicted.reshape(-1, 1) 
    predicted_returns = inverse_scaler.inverse_transform(predicted)
    # test['Predictions'] = predicted_returns

    rmse = mean_squared_error(list(y_test), list(predicted_returns), squared=False)
    print("")
    print("RMSE: ", rmse)


    y_train_df = y_train.to_frame()
    y_train_df.index = pd.to_datetime(y_train_df.index, format='%Y-%m-%d')

    y_test_df = y_test.to_frame()
    y_test_df.columns = ['Actual']

    predicted_returns_df = pd.DataFrame(predicted_returns, index = y_test_df.index, columns=['Predicted'])

    return {
        "train": {
            "dates": list(y_train_df.index.strftime("%Y-%m-%d")),
            "data_points": list(y_train_df['PCT_CHNG'])
        },
        "test": {
            "dates": list(y_test_df.index.strftime("%Y-%m-%d")),
            "data_points": list(y_test_df['Actual'])
        },
        "predicted": {
            "dates": list(predicted_returns_df.index.strftime("%Y-%m-%d")),
            "data_points": list(predicted_returns_df['Predicted'])
        }
    }



if __name__ == '__main__':
    socketio.run(app, port=5300, debug=True)