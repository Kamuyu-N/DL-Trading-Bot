import pandas_ta as ta
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from AssistanceFunctions import DeepLearning
from tensorflow.python.keras.saving.save import load_model

pd.options.display.width = 0

if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()


# change form scaler fit to scaler transform

#Load the model
model = load_model("C:/Users/muyu2/OneDrive/Documents/DeepLearning/models/model_name4.h5")

price = pd.DataFrame(mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M15, 0,50))# problem with this is the timing
price['datetime'] = pd.to_datetime(price['time'], unit='s')
price.drop(['tick_volume', 'spread', 'real_volume','time'], axis=1, inplace=True)
price = price[['datetime', 'open', 'high', 'low', 'close']]
technicals_price = price.copy()
technicals = pd.DataFrame()
technicals = price.drop('datetime', axis = 1)
technicals['ema_12'] = ta.ema(technicals_price['close'],length=12)
technicals['ema_21'] = ta.ema(technicals_price['close'],length=21)
technicals['atr_7'] = ta.atr(technicals_price['high'],technicals_price['low'], technicals_price['close'], length=7)
technicals['atr_14'] = ta.atr(technicals_price['high'],technicals_price['low'], technicals_price['close'], length=14)
technicals.set_index(technicals_price['datetime'], inplace= True)
scaler = MinMaxScaler(feature_range=(-1,1))
temp_df = technicals[['atr_7','atr_14', 'ema_12', 'ema_21']]
log_returns = np.log(temp_df / temp_df.shift(1))
technicals.drop(['atr_7','atr_14', 'ema_12', 'ema_21'], axis=1, inplace=True)
technicals = pd.concat([log_returns,technicals], axis=1)
technicals.dropna(inplace=True, axis= 0)
technicals[['atr_7','atr_14', 'ema_12', 'ema_21']] = scaler.fit_transform(technicals[['atr_7','atr_14', 'ema_12', 'ema_21']])

x1, x2 = DeepLearning.feature_creator(technicals.tail(21))


#Type of trade to take
pred = model.predict([x1, x2])
buy_thresh, sell_thresh = 0.9, 0.9
sl,tp  = 0.0010, 0.0006
avg_spread = 0.00007

symbol = 'EURUSD'
print(pred[0])

#checking the dates for econommic data
#- save the file to storage then parse through ( in a fast manner .. use the hash table ) --VERY IMPORTANT

if pred[0][0] >= buy_thresh:
    price = mt5.symbol_info_tick(symbol).ask

    deviation = 20
    request = {
        "action": mt5.TRADE_ACTION_DEAL,# MARKET ORDER
        "symbol": symbol,
        "volume": 0.6, ## to be reviewed
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl":price - sl,
        "tp": (price + tp) - avg_spread,

    }
    result = mt5.order_send(request)
    print(result)

elif pred[0][1] >= sell_thresh:
    price = mt5.symbol_info_tick(symbol).ask

    deviation = 20
    request = {
        "action": mt5.TRADE_ACTION_DEAL,# MARKET ORDER
        "symbol": symbol,
        "volume": 0.6, ## to be reviewed
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl":price + sl,
        "tp": (price - tp) + avg_spread,

    }
    result = mt5.order_send(request)
    print(result)
