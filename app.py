from flask import Flask, request, jsonify
# from flask_socketio import SocketIO, emit
from flask_cors import CORS

import pandas as pd
import pandas_ta as ta
from pyparsing import Empty
# from pandas_datareader import data
import yfinance as yf
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np

app = Flask(__name__)
CORS(app)

# socketio = SocketIO(app, cors_allowed_origins="*")

df = ""

@app.route("/set_ticker_data", methods = ['POST'])
def send_ticker_data():

    #getting the global dataframe
    global df

    #using data from frontend input
    data = request.get_json()

    # #assigning ticker and time period, 12 refers to 12 months for time_period
    ticker = data['data'].split(" ")[0]
    time_period = data['time_period']

    # #start and end dates
    now = dt.date.today()
    start_date = str(now - relativedelta(months=time_period))
    end_date = str(now)

    # #calling yfinance api to reassigning the global df
    df = yf.download(ticker, start=start_date, end=end_date)

    return_data = {
        "code": 200
    }

    return return_data

@app.route("/get_return_series", methods = ['GET'])
def get_return_series():
    return_df = df.copy()
    dates = list(return_df.index.strftime("%Y-%m-%d"))

    return_df['return_series'] = ((return_df['Adj Close'].pct_change() +1).cumprod() - 1)

    data = return_df['return_series'].dropna()

    data = data.to_list()

    return_series_data = {
        "labels": dates,
        "datasets":[{
            "label": "Adjusted Close Price Return Series",
            "type": 'line',
            "data": data,
            "borderColor": 'rgb(0, 0, 0)',
            "fill": False,
            "borderWidth": 0.5,
            "pointRadius": 1
        }]
    }

    return return_series_data

@app.route("/get_ema", methods = ['GET'])
def getEMAValues():
    return_df = df.copy()

    dates = list(return_df.index.strftime("%Y-%m-%d"))

    return_df['50 day EMA'] = ta.ema(close = return_df['Close'], length = 50)
    return_df['100 day EMA'] = ta.ema(close = return_df['Close'], length = 100)

    def buy_sell(data):

        buy_signal = []
        sell_signal = []
        signal_point = False

        for i in range(len(data)):
            if data['50 day EMA'][i] > data['100 day EMA'][i]:
                if signal_point == False :
                    buy_signal.append(data['Close'][i])
                    # print(data['Close'][i])
                    sell_signal.append(np.nan)
                    signal_point = True #Buy signal as Shorter term EMA > Longer term EMA
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            elif data['50 day EMA'][i] < data['100 day EMA'][i]:
                if signal_point == True:
                    buy_signal.append(np.nan)
                    sell_signal.append(data['Close'][i])
                    signal_point = False #Sell signal as Longer term EMA > Shorter term EMA
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)

        return pd.Series([buy_signal, sell_signal])

    return_df['Buy_Signal_price'], return_df['Sell_Signal_price'] = buy_sell(return_df)

    #Since JSON is unable to take in NaN/missing values, replace all missing values with python's None
    close_price = list(return_df['Close'].replace([np.nan], [None])) 
    ema_50 = list(return_df['50 day EMA'].replace([np.nan], [None]))
    ema_100 = list(return_df['100 day EMA'].replace([np.nan], [None]))
    buy_signal = list(return_df['Buy_Signal_price'].replace([np.nan], [None]))
    sell_signal = list(return_df['Sell_Signal_price'].replace([np.nan], [None]))

    return_data = {
        "labels": dates,
        "datasets":[
            {
                "label": "50 Day EMA",
                "type": 'line',
                "data": ema_50,
                "borderColor": 'rgb(0, 0, 255)',
                "fill": False,
                "borderWidth": 0.5,
                "pointRadius": 1
            },
            {
                "label": "100 Day EMA",
                "type": 'line',
                "data": ema_100,
                "borderColor": 'rgb(255, 0, 132)',
                "fill": False,
                "borderWidth": 0.5,
                "pointRadius": 1
            },
            {
                "label": "Close Price",
                "type": 'line',
                "data": close_price,
                "borderColor": 'rgb(125, 99, 132, 0.5)',
                "fill": False,
                "pointRadius": 1
            },
            {
                "label": "Buy Signal",
                "type": 'line',
                "showLine": False,
                "data": buy_signal,
                "borderColor": 'rgb(0, 255, 0)',
                "fill": False,
                "borderWidth": 5
            },
            {
                "label": "Sell Signal",
                "type": 'line',
                "showLine": False,
                "data": sell_signal,
                "borderColor": 'rgb(255, 0, 0)',
                "fill": False,
                "borderWidth": 5
            },
        ]
    }

    return return_data

@app.route("/get_volatility", methods = ['GET'])
def getVolatilityValues():
    return_df = df.copy()

    dates = list(return_df.index.strftime("%Y-%m-%d"))

    return_df['volatility'] = np.sqrt(252) * pd.DataFrame.rolling(np.log(return_df['Close'] / return_df['Close'].shift(1)),window=20).std()

    #Since JSON is unable to take in NaN/missing values, replace all missing values with python's None
    volatility = list(return_df['volatility'].replace([np.nan], [None])) 

    return_data = {
        "labels": dates,
        "datasets":[
            {
                "label": "20 Day Volatility",
                "type": 'line',
                "data": volatility,
                "borderColor": 'rgb(0, 0, 255)',
                "fill": False,
                "borderWidth": 0.5,
                "pointRadius": 1
            }
        ]
    }

    return return_data

@app.route("/get_macd", methods = ['GET'])
def getMACDValues():
    return_df = df.copy()

    dates = list(return_df.index.strftime("%Y-%m-%d"))

    MACD = ta.macd(close = return_df['Close'])
    return_df = pd.concat([return_df, MACD], axis=1)

    buy_signal=[]
    sell_signal=[]
    signal_point=False

    for i in range(0, len(df)):
        if return_df['MACD_12_26_9'][i] > return_df['MACDs_12_26_9'][i] :
            sell_signal.append(np.nan)
            if signal_point ==False:
                buy_signal.append(return_df['Close'][i])
                signal_point=True
            else:
                buy_signal.append(np.nan)
        elif return_df['MACD_12_26_9'][i] < return_df['MACDs_12_26_9'][i] :
            buy_signal.append(np.nan)
            if signal_point == True:
                sell_signal.append(return_df['Close'][i])
                signal_point=False
            else:
                sell_signal.append(np.nan)
        else:
            buy_signal.append(np.nan)
            sell_signal.append(np.nan)

    return_df['buy_signal_price'] = buy_signal
    return_df['sell_signal_price'] = sell_signal
    
    #Since JSON is unable to take in NaN/missing values, replace all missing values with python's None
    close_price = list(return_df['Close'].replace([np.nan], [None])) 
    macd_values = list(return_df['MACD_12_26_9'].replace([np.nan], [None])) 
    macd_signal_line = list(return_df['MACDs_12_26_9'].replace([np.nan], [None])) 
    macd_histogram_values = list(return_df['MACDh_12_26_9'].replace([np.nan], [None])) 
    buy_signal = list(return_df['buy_signal_price'].replace([np.nan], [None]))
    sell_signal = list(return_df['sell_signal_price'].replace([np.nan], [None]))

    return_data = {
        "labels": dates,
        "datasets":[
            # {
            #     "label": "Close Price",
            #     "type": 'line',
            #     "data": close_price,
            #     "borderColor": 'rgb(0, 0, 255)',
            #     "fill": False,
            #     "borderWidth": 0.5,
            #     "pointRadius": 1
            # },
            {
                "label": "MACD",
                "type": 'line',
                "data": macd_values,
                "borderColor": 'rgb(0, 0, 255)',
                "fill": False,
                "borderWidth": 0.5,
                "pointRadius": 1
            },
            {
                "label": "MACD Histogram",
                "type": 'bar',
                "data": macd_histogram_values,
                "borderColor": 'rgb(0, 0, 255)',
                "fill": False,
                "borderWidth": 0.5,
                "pointRadius": 1
            },
            {
                "label": "MACD Signal Line",
                "type": 'line',
                "data": macd_signal_line,
                "borderColor": 'rgb(0, 255, 0)',
                "fill": False,
                "borderWidth": 0.5,
                "pointRadius": 1
            }
            # },
            # {
            #     "label": "Buy Signal",
            #     "type": 'line',
            #     "showLine": False,
            #     "data": buy_signal,
            #     "borderColor": 'rgb(0, 255, 0)',
            #     "fill": False,
            #     "borderWidth": 4
            # },
            # {
            #     "label": "Sell Signal",
            #     "type": 'line',
            #     "showLine": False,
            #     "data": sell_signal,
            #     "borderColor": 'rgb(255, 0, 0)',
            #     "fill": False,
            #     "borderWidth": 4
            # }
        ]
    }

    return return_data  

@app.route("/get_bollinger", methods=['GET'])
def getBollinger():

    return_df = df.copy()

    dates = list(return_df.index.strftime("%Y-%m-%d"))

    buy_signal = []
    sell_signal = []
    signal_point = False

    bollinger = ta.bbands(return_df['Close'], length=20,std=2)
    return_df = pd.concat([return_df, bollinger], axis=1)

    for i in range(len(return_df)):
        if return_df['Close'][i] < return_df['BBL_20_2.0'][i]:
            if signal_point == False :
                buy_signal.append(return_df['Close'][i])
                sell_signal.append(np.nan)
                signal_point = True
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)
        elif return_df['Close'][i] > return_df['BBU_20_2.0'][i]:
            if signal_point == True:
                buy_signal.append(np.nan)
                sell_signal.append(return_df['Close'][i])
                signal_point = False
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)
        else :
            buy_signal.append(np.nan)
            sell_signal.append(np.nan)

    return_df['bb_Buy_Signal_price'] = buy_signal
    return_df['bb_Sell_Signal_price'] = sell_signal


    #Since JSON is unable to take in NaN/missing values, replace all missing values with python's None
    lower = list(return_df['BBL_20_2.0'].replace([np.nan], [None])) 
    middle = list(return_df['BBM_20_2.0'].replace([np.nan], [None])) 
    upper = list(return_df['BBU_20_2.0'].replace([np.nan], [None])) 
    close_price = list(return_df['Close'].replace([np.nan], [None])) 
    buy_signal = list(return_df['bb_Buy_Signal_price'].replace([np.nan], [None]))
    sell_signal = list(return_df['bb_Sell_Signal_price'].replace([np.nan], [None]))

    return_data = {
        "labels": dates,
        "datasets":[
            {
                "label": "Bollinger Lower",
                "type": 'line',
                "data": lower,
                "borderColor": 'rgb(0, 0, 255)',
                "fill": False,
                "borderWidth": 0.5,
                "pointRadius": 1
            },
            {
                "label": "Bollinger Middle",
                "type": 'line',
                "data": middle,
                "borderColor": 'rgb(252, 208, 23)',
                "fill": False,
                "borderWidth": 0.5,
                "pointRadius": 1
            },
            {
                "label": "Bollinger Upper",
                "type": 'line',
                "data": upper,
                "borderColor": 'rgb(255, 0, 0)',
                "fill": False,
                "borderWidth": 0.5,
                "pointRadius": 1
            },
            {
                "label": "Close Price",
                "type": 'line',
                "data": close_price,
                "borderColor": 'rgb(0, 0, 0)',
                "fill": False,
                "borderWidth": 0.5,
                "pointRadius": 1
            },
            {
                "label": "Buy Signal",
                "type": 'line',
                "showLine": False,
                "data": buy_signal,
                "borderColor": 'rgb(0, 255, 0)',
                "fill": False,
                "borderWidth": 4
            },
            {
                "label": "Sell Signal",
                "type": 'line',
                "showLine": False,
                "data": sell_signal,
                "borderColor": 'rgb(255, 0, 0)',
                "fill": False,
                "borderWidth": 4
            }
        ]
    }

    return return_data
# def get_tickers_data(tickers, time_period):
#     now = dt.date.today()
#     start_date = str(now - relativedelta(months=time_period))
#     end_date = str(now)

#     df = yf.download(tickers, start=start_date, end=end_date)

#     try:
#         df = yf.download(tickers, start=start_date, end=end_date)

#         if len(tickers) == 1:
#             close_price = list(df['Close'])
#         elif len(tickers) > 1:
#             close_price = list(df['Close'][tickers[0]])

#         dates = list(df.index.strftime("%Y-%m-%d"))
#         return_data = {
#             "dates": dates, 
#             "close_price": close_price
#         }
#         return return_data
#     except Exception as e:
#         print("error! failed to get Yahoo data")
#         print(e)
#         return "Failed"



# @socketio.on('Slider value changed')
# def value_changed(message):
#     values[message['who']] = message['data']
#     emit('update value', message, broadcast=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5100, debug=True)