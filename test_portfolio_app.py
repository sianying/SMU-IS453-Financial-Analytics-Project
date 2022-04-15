from mimetypes import init
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
import scipy.optimize as sco
app = Flask(__name__)
CORS(app)

# socketio = SocketIO(app, cors_allowed_origins="*")

df = ""
risk_level = ""
user_ticker = ""
time_period =""
@app.route("/set_ticker_data", methods = ['POST'])
def send_ticker_data():

    #getting the global dataframe
    global df
    global risk_level
    global user_ticker
    global time_period
    #using data from frontend input
    data = request.get_json()
    print(data)
    # #assigning ticker and time period, 12 refers to 12 months for time_period
    ticker = data['data']
    time_period = int(data['time_period'])
    risk_level = data['risk']
    user_tickers = [ticker.split(" ")[0] for ticker in ticker]
    # #start and end dates
    now = dt.date.today()   
    start_date = str(now - relativedelta(months=time_period))
    end_date = str(now)


    # #calling yfinance api to reassigning the global df
    df = yf.download(user_tickers, start=start_date, end=end_date)
    print(type(ticker))
    user_ticker = user_tickers


    return_data = {
        "code": 200
    }

    return return_data

@app.route("/get_po", methods = ['GET'])
def get_po():
    return_df = df.copy()

    data = "123"
    df_closes = return_df['Adj Close']
    close_price = return_df['Close']
    returns = df_closes.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_portfolios = 3000
    risk_free_rate = 0.0283 # US Treasury 10 Year Rate
    adj_close = return_df["Adj Close"]
    returns_ts = adj_close.pct_change().dropna()
    avg_daily_ret = returns_ts.mean()
    returns_ts['RiskFree_Rate'] = risk_free_rate/252
    avg_rf_ret = returns_ts['RiskFree_Rate'].mean()
    ret_series_close = []
    return_series_close = (close_price.pct_change()+ 1).cumprod() - 1 
    return_series_close = return_series_close.dropna()
    for funds in user_ticker:
        ret_series_close.append(return_series_close[funds].values.tolist())

    funds = return_series_close.columns.values.tolist()
    weights_record = []
    ef_list = []
   
    all_weights = np.zeros((num_portfolios, len(user_ticker)))
    ret_arr = np.zeros(num_portfolios) # Returns Array
    vol_arr = np.zeros(num_portfolios) # Volatility Array
    sharpe_arr = np.zeros(num_portfolios) # SHarpe Array
    #Simulating 3000 portfolios and appending the results to the respective arrays
    for i in range(num_portfolios):
            weights = np.random.random(len(user_ticker)) # Generating random weights based on number of funds
            weights /= np.sum(weights) #Ensure all the weights sums up to 1
            weights_record.append(np.round(weights,2)) # Storing all weights records
            all_weights[i,:] = weights # Storing all weights records
            ret_arr[i] = np.sum( (mean_returns * weights * 252)) # Calculating returns and adding it to the array
            vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix*252, weights))) #Calculating volatility and adding it to the array
            sharpe_arr[i] = ret_arr[i]/vol_arr[i] # Calculating sharpe ratio and adding it to the array
            ef_list.append({"x": vol_arr[i], "y": ret_arr[i]}) #EF_List is to store all portfolios for plotting

    max_ret = ret_arr.max() # Getting the max returns
    min_ret = ret_arr.min() # Getting the minimum returns
    
    # Calculating returns and volatility given weights
    def get_ret_vol_sr(weights):
        weights = np.array(weights)
        ret = np.sum((mean_returns*weights ) *252)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return np.array([ret, vol])
    # Ensure that the weights summed up to 1
    def check_sum(weights):
        #return 0 if sum of the weights is 1
        return np.sum(weights)-1
    
    cons = ({'type': 'eq', 'fun': check_sum})
    bounds = ()
    for i in range(len(user_ticker)):
        bounds = bounds + ((0,1),)
    no_of_assets = len(user_ticker)

    frontier_y = np.linspace(min_ret,max_ret,500)   

    def minimize_volatility(weights):
        return get_ret_vol_sr(weights)[1]
    list_vol = [] #volatility
    list_ret = [] #return
    list_weight = [] # Weights
    list_sr = [] # Sharpe Ratio

    # Looping through possible returns and minimizing the volatility for each returns using scipy's minimize function
    for possible_return in frontier_y:
        cons = ({'type': 'eq', 'fun': check_sum},
                {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return}
                )
        a_result = sco.minimize(minimize_volatility,no_of_assets*[1./no_of_assets,],method="SLSQP", bounds = bounds, constraints = cons)
        list_vol.append(a_result['fun'])
        list_ret.append(possible_return)
        list_sr.append((possible_return - risk_free_rate)/a_result['fun'])
        list_weight.append(a_result['x'])
    
    # list_x_to_arr to sort the array in ascending order to get the max values
    list_vol_to_arr = list_vol.copy()
    list_vol_to_arr.sort()
    list_sr_to_arr = list_sr.copy()
    list_sr_to_arr.sort()

    low_risk_vol = list_vol_to_arr[33] 
    med_risk_vol = list_vol_to_arr[66]
    high_risk_vol = list_vol_to_arr[-1]
    # Getting the index of the respective volatility in order to extract out its corresponding returns, volatility, and weights
    for i in range(len(list_vol)):
        if(low_risk_vol == list_vol[i]):
            low_risk_vol_idx = i
        elif(med_risk_vol == list_vol[i]):
            med_risk_vol_idx = i
        elif(high_risk_vol == list_vol[i]):
            high_risk_vol_idx = i

    low_risk_ret = list_ret[low_risk_vol_idx] # Returns of the low risk volatility
    med_risk_ret = list_ret[med_risk_vol_idx] # Returns of the med risk volatility
    high_risk_ret = list_ret[high_risk_vol_idx] # Returns of the high risk volatility
    low_risk_ret_point = [{"x": low_risk_vol, "y": low_risk_ret}] # To be returned for plotting of low risk point on graph
    med_risk_ret_point = [{"x": med_risk_vol, "y": med_risk_ret}] # To be returned for plotting of med risk point on graph
    high_risk_ret_point = [{"x": high_risk_vol, "y": high_risk_ret}] # To be returned for plotting of high risk point on graph
    low_risk_sr = (low_risk_ret - risk_free_rate) / low_risk_vol # Sharpe ratio of the low risk point
    med_risk_sr = (med_risk_ret - risk_free_rate) / med_risk_vol # Sharpe ratio of the med risk point
    high_risk_sr = (high_risk_ret - risk_free_rate) / high_risk_vol # Sharpe ratio of the high risk point

    low_risk_weight = list_weight[low_risk_vol_idx] # Weight of the low risk point
    med_risk_weight = list_weight[med_risk_vol_idx] # Weight of the med risk point
    high_risk_weight = list_weight[high_risk_vol_idx] # Weight of the high risk point

    # Evaluating the Sharpe Ratio
    low_rar_evaluation = ""
    if (low_risk_sr < 1):
        low_rar_evaluation = "Not Good"
    elif (1 <= low_risk_sr <= 1.99):
        low_rar_evaluation= "Acceptable - Good"
    elif (2 <= low_risk_sr <= 2.99):
        low_rar_evaluation = "Very Good"
    else:
        low_rar_evaluation = "Great"


    med_rar_evaluation = ""
    if (med_risk_sr < 1):
        med_rar_evaluation = "Not Good"
    elif (1 <= med_risk_sr <= 1.99):
        med_rar_evaluation= "Acceptable - Good"
    elif (2 <= med_risk_sr <= 2.99):
        med_rar_evaluation = "Very Good"
    else:
        med_rar_evaluation = "Great"

    high_rar_evaluation = ""
    if (high_risk_sr < 1):
        high_rar_evaluation = "Not Good"
    elif (1 <= high_risk_sr <= 1.99):
        high_rar_evaluation = "Acceptable - Good"
    elif (2 <= high_risk_sr <= 2.99):
        high_rar_evaluation = "Very Good"
    else:
        high_rar_evaluation = "Great"

    
    list_ret_to_arr = list_ret.copy()
    list_ret_to_arr.sort()
    high_risk_ret = list_ret_to_arr[-1]

    # Efficient Frontier Line
    best_ef_list = []
    for e in range(len(list_vol)):
        best_ef_list.append({"x": list_vol[e], "y": frontier_y[e]})

    dates = list(df.index.strftime("%Y-%m-%d"))
    if risk_level == 'Low':
        data = {
            "dates": dates, 
            "ticker": funds,
            "combi" : list(low_risk_weight),
            "ef_list": ef_list,
            "best_ef_list": best_ef_list,
            "risk_level_point" : low_risk_ret_point,
            "return_series" : ret_series_close,
            "volatility": round(low_risk_vol,2),
            "returns": round(low_risk_ret,2),
            "rar": round(low_risk_sr,20),
            "rar_evaluation": low_rar_evaluation
        }
        return data
    elif risk_level == 'Medium':
        data = {
            "dates": dates, 
            "ticker": funds,
            "combi" : list(med_risk_weight),
            "ef_list": ef_list,
            "best_ef_list": best_ef_list,
            "risk_level_point" : med_risk_ret_point,
            "return_series" : ret_series_close,
            "volatility": round(med_risk_vol,2),
            "returns": round(med_risk_ret,2),
            "rar": round(med_risk_sr,2),
            "rar_evaluation": med_rar_evaluation
        }
        return data
    elif risk_level == 'High':
        data = {
            "dates": dates, 
            "ticker": funds,
            "combi" : list(high_risk_weight),
            "ef_list": ef_list,
            "best_ef_list": best_ef_list,
            "risk_level_point" : high_risk_ret_point,
            "return_series" : ret_series_close,
            "volatility": round(high_risk_vol,2),
            "returns": round(high_risk_ret,2),
            "rar": round(high_risk_sr,2),
            "rar_evaluation": high_rar_evaluation
        }
        return data



    


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5200, debug=True)