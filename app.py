from flask import Flask, render_template
from flask_socketio import SocketIO, emit

import pandas as pd
import pandas_datareader.data as web
import datetime as dt

app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")

# @socketio.on('my event')
# def handle_message(data):
#     print('received message: ' + str(data))
#     return "acknowledged!"

@socketio.on('ticker-data')
def send_ticket_data(client_data):

    print(client_data)
    tickers_input = client_data['data']
    time_period = client_data['time_period']

    tickers = [ticker.split(" ")[0] for ticker in tickers_input]
    print(tickers)

    data = get_tickers_data(tickers, time_period)
    print(data)

    #convert data to dict
    # dates = df.drop(columns=['Close', 'Adj Close', 'High', 'Low', 'Open', 'Volume'])
    # print(df[dates])
    # close_price = df['Close']
    # print(close_price)

    return {
        "code": "received!",
        "data": data
    }

def get_tickers_data(tickers, time_period):
    now = dt.date.today()
    start = now - dt.timedelta(days=time_period)
    end = now

    try:
        df = web.DataReader(tickers, 'yahoo', start, end)
        # df[df.index.duplicated(keep='first')]
        close_price = list(df['Close'][tickers[0]])
        dates = list(df.index.strftime("%Y-%m-%d"))
        data = {
            "dates": dates, 
            "close_price": close_price
        }
        return data
    except:
        print("error! failed to get Yahoo data")
        return "Failed! :("



# @socketio.on('Slider value changed')
# def value_changed(message):
#     values[message['who']] = message['data']
#     emit('update value', message, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, port=5100, debug=True)