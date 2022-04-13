import requests
import re
import datetime as dt
import pandas as pd

import snscrape.modules.twitter as sntwitter
from pandas_datareader import data

def etf_holdings_weights(etf, top_holdings=5):
    '''
        Input: 
            - etf: <string> etf to scrape
            - top_holdings: <integer> top XX holdings (default is top 5)

        Return: <dict> top XX ETF's holdings and their weights
    '''
    url="https://www.zacks.com/funds/etf/{}/holding"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"
    }

    with requests.Session() as req:
        req.headers.update(headers)

        r = req.get(url.format(etf))
        # print(f"Extracting: {r.url}")

        #scrapes for holdings of the etf
        holdings = re.findall(r'etf\\\/(.*?)\\', r.text)[:5]

        #scrapes weights but with html attributes
        weights_result = re.findall(r'<\\\/span><\\\/span><\\\/a>",(.*?), "<a class=\\\"report_document newwin\\', r.text)
        weights = [weights_result[i].split()[1][1:-2] for i in range(len(weights_result)) if i < top_holdings]
        
        return dict(zip(holdings, weights))

def twitter_scraper(query_list, num_days):
    '''
        Input:
            - query_list: <list> list of etf + its holdings e.g. ['$VGT', '$APPL']
            - num_days: <integre> number of days to web scrape
        Return:
            - tweets_df: <df> df of scraped tweets
    '''
    start_date = dt.date.today() - dt.timedelta(days=num_days)
    end_date = dt.date.today()

    #add $ if stock len is 2 and below e.g. Ma, V -> $MA, $V
    query_list_adj = []
    for stock in query_list:
        if len(stock) <= 2:
            stock = f'${stock}'
            query_list_adj.append(stock)
        else:
            query_list_adj.append(stock)

    print(query_list_adj)

    query = ' OR '.join(query_list_adj) # '$VGT OR $APPL'
    query_statement = f'{query} lang:en since:{start_date} until:{end_date}'

    tweets = []

    #Using TwitterSearchScraper to scrape data and append tweets to list
    for i ,tweet in enumerate(sntwitter.TwitterSearchScraper(query_statement).get_items()):
        tweets.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    tweets_df = pd.DataFrame(tweets, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
    # tweets_df.to_csv(f"VGT_tweets_{start_date[0:7]}_{end_date[0:7]}.csv")
    return tweets_df