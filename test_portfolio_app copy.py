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
@app.route("/set_ticker_data", methods = ['POST'])
def send_ticker_data():

    #getting the global dataframe
    global df
    global risk_level
    global user_ticker
    #using data from frontend input
    data = request.get_json()
    #print(data)
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
    #print(type(ticker))
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
    log_ret = np.log(df_closes/df_closes.shift(1))
    returns = df_closes.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_portfolios = 3000
    risk_free_rate = 0.01 #0.0178
    adj_close = return_df["Adj Close"]
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
    an_vol = np.std(returns) * np.sqrt(252) #volatility of funds
    an_rt = mean_returns * 252
    results = np.zeros((3,num_portfolios))
    weights_record = []
    ef_list = []
   
    all_weights = np.zeros((num_portfolios, len(user_ticker)))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)
    for i in range(num_portfolios):
            weights = np.random.random(len(user_ticker))
            weights /= np.sum(weights)
            weights_record.append(np.round(weights,2))
            all_weights[i,:] = weights
            ret_arr[i] = np.sum( (log_ret.mean() * weights * 252))
            vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))
            sharpe_arr[i] = ret_arr[i]/vol_arr[i]
            ef_list.append({"x": vol_arr[i], "y": ret_arr[i]})
    
    max_sharpe_ratio = sharpe_arr.max()
    max_sharpe_ratio_idx = sharpe_arr.argmax()
    max_sr_ret = ret_arr[max_sharpe_ratio_idx]
    max_sr_vol = vol_arr[max_sharpe_ratio_idx]

    max_sharpe_ratio_point = [{"x": max_sr_vol, "y": max_sr_ret}]
    
    min_vol_vol = vol_arr[vol_arr.argmin()]
    min_vol_ret = ret_arr[vol_arr.argmin()]
    min_vol_sharpe_ratio = sharpe_arr[vol_arr.argmin()]
    min_vol_sharpe_ratio_point = [{"x": min_vol_vol, "y": min_vol_ret}]

    max_ret = ret_arr.max()
    print("this is max ret")
    print(max_ret)
    min_ret = ret_arr.min()
    

    def get_ret_vol_sr(weights):
        weights = np.array(weights)
        ret = np.sum((mean_returns*weights ) *252)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sr = ret/vol
        return np.array([ret, vol, sr])

    def neg_sharpe(weights):
    # the number 2 is the sharpe ratio index from the get_ret_vol_sr
        return get_ret_vol_sr(weights)[2] * -1

    def check_sum(weights):
        #return 0 if sum of the weights is 1
        return np.sum(weights)-1
    
    cons = ({'type': 'eq', 'fun': check_sum})
    bounds = ()
    for i in range(len(user_ticker)):
        bounds = bounds + ((0,1),)
    #bounds = ((0,1),(0,1),(0,1),(0,1))
    no_of_assets = len(user_ticker)
    #opt_results = sco.minimize(neg_sharpe,init_guess,method="SLSQP", bounds = bounds, constraints = cons)
    #print(opt_results)
    frontier_y = np.linspace(min_ret,max_ret,500)
    def minimize_volatility(weights):
        return get_ret_vol_sr(weights)[1]
    list_vol = [] #volatility
    list_ret = [] #return
    list_weight = []
    for possible_return in frontier_y:
        cons = ({'type': 'eq', 'fun': check_sum},
                {'type': 'eq', 'fun': lambda w: get_ret_vol_sr(w)[0] - possible_return}
                )
        a_result = sco.minimize(minimize_volatility,no_of_assets*[1./no_of_assets,],method="SLSQP", bounds = bounds, constraints = cons)
        list_vol.append(a_result['fun'])
        list_ret.append(possible_return)
        list_weight.append(a_result['x'])
    
    list_vol_to_arr = list_vol.copy()
    list_vol_to_arr.sort()
    print("this is max vol")
    print(list_vol_to_arr[-1])
    low_risk_vol = list_vol_to_arr[0]
    med_risk_vol = list_vol_to_arr[24]
    for i in range(len(list_vol)):
        if(low_risk_vol == list_vol[i]):
            low_risk_vol_idx = i
        elif(med_risk_vol == list_vol[i]):
            med_risk_vol_idx = i
    low_risk_ret = list_ret[low_risk_vol_idx]
    med_risk_ret = list_ret[med_risk_vol_idx]
    low_risk_ret_point = [{"x": low_risk_vol, "y": low_risk_ret}]
    low_risk_sr = (low_risk_ret - risk_free_rate) / low_risk_vol
    med_risk_sr = (med_risk_ret - risk_free_rate) / med_risk_vol
    med_risk_ret_point = [{"x": med_risk_vol, "y": med_risk_ret}]
    low_risk_weight = list_weight[low_risk_vol_idx]

    med_risk_weight = list_weight[med_risk_vol_idx]
    list_ret_to_arr = list_ret.copy()
    list_ret_to_arr.sort()
    high_risk_ret = list_ret_to_arr[27]
    for i in range (len(list_ret)):
        if(high_risk_ret == list_ret[i]):
            high_risk_vol_idx = i
    high_risk_vol = list_vol[high_risk_vol_idx]
    high_risk_sr = (high_risk_ret - risk_free_rate) / high_risk_vol
    high_risk_weight = list_weight[high_risk_vol_idx]
    high_risk_ret_point = [{"x": high_risk_vol, "y": high_risk_ret}]
    best_ef_list = []
    for e in range(len(list_vol)):
        best_ef_list.append({"x": list_vol[e], "y": frontier_y[e]})

    print(low_risk_vol,type(low_risk_vol))
    print(low_risk_ret, type(low_risk_ret))
    print(low_risk_sr, type(low_risk_sr))
    

   # print("--------"*80)
   # print(frontier_x[1],frontier_y[1])

            #returns = np.sum(mean_returns*weights ) *252

            # std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            # portfolio_std_dev, portfolio_return = std, returns
            # results[0,i] = std #volatility
            # results[1,i] = returns #returns
            # results[2,i] = (returns - risk_free_rate) / std #sharpe ratio
            # ef_list.append({"x":results[0,i], "y": results[1,i]}) #Efficient Frontier List to be returned in the chartjs format
            
    #max_sharpe_idx = np.argmax(results[2]) #Finding the max sharpe idx #
    #sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx] #sdp => volatility of the max sharpe, rp => returns of the max sharpe
    # max_sharpe_allocation = pd.DataFrame(weights_record[max_sharpe_idx],index=df_closes.columns,columns=['allocation']) #Finding the weights_records aka combination
    # max_sharpe_allocation.allocation = [round(i,2)for i in max_sharpe_allocation.allocation]
    # max_sharpe_allocation = max_sharpe_allocation.T 

    # min_vol_idx = np.argmin(results[0])
    # sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    # min_vol_allocation = pd.DataFrame(weights_record[min_vol_idx],index=df_closes.columns,columns=['allocation'])
    # min_vol_allocation.allocation = [round(i,2)for i in min_vol_allocation.allocation]
    # min_vol_allocation = min_vol_allocation.T
    


    # results1 = results[:, results[0, :].argsort()] #sort according to volatility in ascending order
    # #LOW RISK
    # low_risk_rar = results1[2,999] #Getting the RAR value for low ris
    # tuple_low_risk_weight_idx = np.where(results[2] == low_risk_rar) # Find index from initial results list where RAR equals to low risk rar
    # print("SD", results[0,999], type(results[0,999]))
    # print("RET", results[1,999], type(results[1,999]))
    # low_risk_point = [{"x": results[0,999], "y": results[1,999]}]
    # new_tuple_low_risk = tuple_low_risk_weight_idx[0].astype(int)
    # low_risk_combi = weights_record[new_tuple_low_risk[0]] #LOW RISK WEIGHT COMBI
    # low_rar_evaluation = ""
    # if (low_risk_rar < 1):
    #     low_rar_evaluation = "Not Good"
    # elif (1 <= low_risk_rar <= 1.99):
    #     low_rar_evaluation= "Acceptable - Good"
    # elif (2 <= low_risk_rar <= 2.99):
    #     low_rar_evaluation = "Very Good"
    # else:
    #     low_rar_evaluation = "Great"

    # med_risk_rar = results1[2, 1999]
    # tuple_med_risk_weight_idx = np.where(results[2] == med_risk_rar) #finding the weight composition by finding the index first
    # new_tuple_med_risk = tuple_med_risk_weight_idx[0].astype(int)
    # med_risk_combi = weights_record[new_tuple_med_risk[0]] #med RISK WEIGHT COMBI

    # med_rar_evaluation = ""
    # if (med_risk_rar < 1):
    #     med_rar_evaluation = "Not Good"
    # elif (1 <= med_risk_rar <= 1.99):
    #     med_rar_evaluation= "Acceptable - Good"
    # elif (2 <= med_risk_rar <= 2.99):
    #     med_rar_evaluation = "Very Good"
    # else:
    #     med_rar_evaluation = "Great"



    # high_risk_rar = results1[2,2999]
    # tuple_high_risk_weight_idx = np.where(results[2] == high_risk_rar) #finding the weight composition by finding the index first
    # new_tuple_high_risk = tuple_high_risk_weight_idx[0].astype(int)
    # high_risk_combi = weights_record[new_tuple_high_risk[0]] #med RISK WEIGHT COMBI

    # high_rar_evaluation = ""
    # if (high_risk_rar < 1):
    #     high_rar_evaluation = "Not Good"
    # elif (1 <= high_risk_rar <= 1.99):
    #     high_rar_evaluation= "Acceptable - Good"
    # elif (2 <= high_risk_rar <= 2.99):
    #     high_rar_evaluation = "Very Good"
    # else:
    #     high_rar_evaluation = "Great"

    #print(new_tuple_low_risk[0], type(new_tuple_low_risk[0]),"<<<<<<<<<<<<<<<<<<")
    # print("weight composition", weights_record[new_tuple_low_risk[0]], type(weights_record[new_tuple_low_risk[0]] ))
    # print("HERE"*40)
    # print("low risk",low_risk_rar, type(low_risk_rar))
    
    # max_sharpe_weight = list(weights_record[max_sharpe_idx])
    # min_vol_weight = list(weights_record[min_vol_idx])
    # max_sharpe_value = results[2,max_sharpe_idx].astype(float)
    # min_vol_sharpe_value = results[2,min_vol_idx].astype(float)
    # list_an_rt = list(an_rt)
    # list_rt_vol = []
    # for i in range(len(list_an_rt)):
    #     list_rt_vol.append({"x":an_vol[i], "y": an_rt[i]})
    
    # #print( "list", type(low_risk_rar), "Low risk combi", type(low_risk_combi))
    # low_risk_combination = list(low_risk_combi)
    # med_risk_combination = list(med_risk_combi)
    # high_risk_combination = list(high_risk_combi)
    # max_sharpe_ratio = [{"x": sdp, "y": rp}] # Used to plot the max sharpe ratio on EF graph 
    # min_vol_ratio = [{"x" : sdp_min, "y": rp_min}] # Used to plot the min vol ratio on EF graph

    dates = list(df.index.strftime("%Y-%m-%d"))
    print("MED RISK WEIGHT")
    print(high_risk_weight)
    if risk_level == 'Low':
        data = {
            "dates": dates, 
            "ticker": user_ticker,
            "combi" : list(low_risk_weight),
            "ef_list": ef_list,
            "best_ef_list": best_ef_list,
            "risk_level_point" : low_risk_ret_point
        }
        return data
    elif risk_level == 'Medium':
        data = {
            "dates": dates, 
            "ticker": user_ticker,
            "combi" : list(med_risk_weight),
            "ef_list": ef_list,
            "best_ef_list": best_ef_list,
            "risk_level_point" : med_risk_ret_point
        }
        return data
    elif risk_level == 'High':
        data = {
            "dates": dates, 
            "ticker": user_ticker,
            "combi" : list(high_risk_weight),
            "ef_list": ef_list,
            "best_ef_list": best_ef_list,
            "risk_level_point" : high_risk_ret_point
        }
        return data



    


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5200, debug=True)