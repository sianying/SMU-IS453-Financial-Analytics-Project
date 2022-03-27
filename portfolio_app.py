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
risk_level = ""
user_ticker = ""
@app.route("/set_ticker_data", methods = ['POST'])
def send_ticker_data():

    #getting the global dataframe
    global df
    global risk_level
    global user_ticker
    #using data from frontend input
    data = request.get_json()
    print(data)
    # #assigning ticker and time period, 12 refers to 12 months for time_period
    ticker = data['data']
    time_period = data['time_period']
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
    #dates = list(return_df.index.strftime("%Y-%m-%d"))
    data = "123"
    
    #print(user_ticker)
    #print(type(user_ticker))
    #print(return_df)

    
    #return_df['return_series'] = ((return_df['Adj Close'].pct_change() +1).cumprod() - 1)
    #data = return_df['return_series'].dropna()
    #data = data.to_list()
    df_closes = return_df['Adj Close']
    returns = df_closes.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_portfolios = 3000
    risk_free_rate = 0.01 #0.0178
    adj_close = return_df["Adj Close"]
    risk_free_ann_ret_rate = 0.01 #to be changed
    returns_ts = adj_close.pct_change().dropna()
    avg_daily_ret = returns_ts.mean()
    returns_ts['RiskFree_Rate'] = risk_free_rate/252
    avg_rf_ret = returns_ts['RiskFree_Rate'].mean()

    # calculate individualsharpe ratio   
    ind_sharpe = []
    for i in range (len(user_ticker)):
    #Add the excess return columns for each ETF
        returns_ts['Excess_ret_' + user_ticker[i]] = returns_ts[user_ticker[i]] - returns_ts['RiskFree_Rate']
        sharpe = ((avg_daily_ret[user_ticker[i]] - avg_rf_ret) /returns_ts['Excess_ret_' + user_ticker[i]].std())*np.sqrt(252)
        ind_sharpe.append(sharpe)
    an_vol = np.std(returns) * np.sqrt(252)
    an_rt = mean_returns * 252
    results = np.zeros((3,num_portfolios))
    weights_record = []
    ef_list = []
    for i in range(num_portfolios):
            weights = np.random.random(len(user_ticker))
            weights /= np.sum(weights)
            weights_record.append(np.round(weights,2))
            returns = np.sum(mean_returns*weights ) *252

            std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            portfolio_std_dev, portfolio_return = std, returns
            results[0,i] = std
            results[1,i] = returns
            results[2,i] = (returns - risk_free_rate) / std #sharpe ratio
            ef_list.append({"x":results[0,i], "y": results[1,i]})
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights_record[max_sharpe_idx],index=df_closes.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights_record[min_vol_idx],index=df_closes.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    results1 = results[:, results[0, :].argsort()]
    #LOW RISK
    low_risk_rar = results1[2,999]
    tuple_low_risk_weight_idx = np.where(results[2] == low_risk_rar) #finding the weight composition by finding the index first
    new_tuple_low_risk = tuple_low_risk_weight_idx[0].astype(int)
    low_risk_combi = weights_record[new_tuple_low_risk[0]] #LOW RISK WEIGHT COMBI
    low_rar_evaluation = ""
    if (low_risk_rar < 1):
        low_rar_evaluation = "Not Good"
    elif (1 <= low_risk_rar <= 1.99):
        low_rar_evaluation= "Acceptable - Good"
    elif (2 <= low_risk_rar <= 2.99):
        low_rar_evaluation = "Very Good"
    else:
        low_rar_evaluation = "Great"

    med_risk_rar = results1[2, 1999]
    tuple_med_risk_weight_idx = np.where(results[2] == med_risk_rar) #finding the weight composition by finding the index first
    new_tuple_med_risk = tuple_med_risk_weight_idx[0].astype(int)
    med_risk_combi = weights_record[new_tuple_med_risk[0]] #med RISK WEIGHT COMBI

    med_rar_evaluation = ""
    if (med_risk_rar < 1):
        med_rar_evaluation = "Not Good"
    elif (1 <= med_risk_rar <= 1.99):
        med_rar_evaluation= "Acceptable - Good"
    elif (2 <= med_risk_rar <= 2.99):
        med_rar_evaluation = "Very Good"
    else:
        med_rar_evaluation = "Great"



    high_risk_rar = results1[2,2999]
    tuple_high_risk_weight_idx = np.where(results[2] == high_risk_rar) #finding the weight composition by finding the index first
    new_tuple_high_risk = tuple_high_risk_weight_idx[0].astype(int)
    high_risk_combi = weights_record[new_tuple_high_risk[0]] #med RISK WEIGHT COMBI

    high_rar_evaluation = ""
    if (high_risk_rar < 1):
        high_rar_evaluation = "Not Good"
    elif (1 <= high_risk_rar <= 1.99):
        high_rar_evaluation= "Acceptable - Good"
    elif (2 <= high_risk_rar <= 2.99):
        high_rar_evaluation = "Very Good"
    else:
        high_rar_evaluation = "Great"

    #print(new_tuple_low_risk[0], type(new_tuple_low_risk[0]),"<<<<<<<<<<<<<<<<<<")
    print("weight composition", weights_record[new_tuple_low_risk[0]], type(weights_record[new_tuple_low_risk[0]] ))
    print("HERE"*40)
    print("low risk",low_risk_rar, type(low_risk_rar))
    
    max_sharpe_weight = list(weights_record[max_sharpe_idx])
    min_vol_weight = list(weights_record[min_vol_idx])
    max_sharpe_value = results[2,max_sharpe_idx].astype(float)
    min_vol_sharpe_value = results[2,min_vol_idx].astype(float)
    list_an_rt = list(an_rt)
    list_an_vol = list(an_vol)
    list_rt_vol = []
    for i in range(len(list_an_rt)):
        list_rt_vol.append({"x":an_vol[i], "y": an_rt[i]})
    
    print( "list", type(low_risk_rar), "Low risk combi", type(low_risk_combi))
    low_risk_combination = list(low_risk_combi)
    med_risk_combination = list(med_risk_combi)
    high_risk_combination = list(high_risk_combi)
    max_sharpe_ratio = [{"x": sdp, "y": rp}]
    min_vol_ratio = [{"x" : sdp_min, "y": rp_min}]

    dates = list(df.index.strftime("%Y-%m-%d"))
    
    if risk_level == 'Low':
        data = {
            "dates": dates, 
            "ticker": user_ticker,
            "ind_sharpe": ind_sharpe,
            "max_sharpe_ratio": max_sharpe_ratio,
            "min_vol_ratio": min_vol_ratio,
            "combi" : low_risk_combination,
            "rar" : round(low_risk_rar,2),
            "allocation": min_vol_weight,
            "sharpe": min_vol_sharpe_value,
            "ef_list": ef_list,
            "list_rt_vol": list_rt_vol,
            "rar_evaluation" : low_rar_evaluation
        }
        return data
    elif risk_level == 'Medium':
        data = {
            "dates": dates, 
            "ticker": user_ticker,
            "ind_sharpe": ind_sharpe,
            "max_sharpe_ratio": max_sharpe_ratio,
            "min_vol_ratio": min_vol_ratio,
            "combi" : med_risk_combination,
            "rar" : round(med_risk_rar,2),
            "allocation": min_vol_weight,
            "sharpe": min_vol_sharpe_value,
            "ef_list": ef_list,
            "list_rt_vol": list_rt_vol,
            "rar_evaluation" : med_rar_evaluation
        }
        return data
    elif risk_level == 'High':
        data = {
            "dates": dates, 
            "ticker": user_ticker,
            "ind_sharpe": ind_sharpe,
            "max_sharpe_ratio": max_sharpe_ratio,
            "min_vol_ratio": min_vol_ratio,
            "combi" : high_risk_combination,
            "rar" : round(high_risk_rar,2),
            "allocation": max_sharpe_weight,
            "sharpe":max_sharpe_value,
            "ef_list": ef_list,
            "list_rt_vol": list_rt_vol,
            "rar_evaluation" : high_rar_evaluation
        }
        return data



    


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5200, debug=True)