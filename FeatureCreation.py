import pandas as pd
import numpy as np
import pandas_ta as ta
from AssistanceFunctions import DeepLearning
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

pd.options.display.width = 0

#import the data
price = pd.read_csv('C:/Users/muyu2/OneDrive/Documents/DeepLearning/eurusd-15m.csv', sep=';')

price.columns = ['date', 'time', 'open', 'high','low','close', 'volume']
price['datetime'] = price['date'] + " "+ price['time']
price.drop(['date', 'time'], axis=1, inplace=True)

#Change column arrangement
price = price[['datetime', 'open','high','low','close', 'volume']]
price.drop('volume', axis= 1, inplace=True)

#Create 30 min timeframe data
technicals_price = price.copy()
technicals = price.copy()

#Create log prices
cols = ['open', 'high', 'low', 'close']
for col in cols:
    technicals[f'log_{col}'] = np.log(technicals[col] / technicals[col].shift(1))

#Feature creation
technicals['ema_mid'] = ta.ema(technicals['close'], length=100)
technicals['ema_long'] = ta.ema(technicals['close'], length=200)
technicals['ema_short'] = ta.ema(technicals['close'], length=50)
technicals['atr'] = ta.atr(technicals['high'], technicals['low'], technicals['close'], length=14)

technicals.dropna(axis = 0, inplace = True)

#Calculate the distance between price and current moving average
technicals['ema_short'] = (technicals['ema_short'] - technicals['close'])
technicals['ema_mid'] = (technicals['ema_mid'] - technicals['close'])
technicals['ema_long']= (technicals['ema_long'] - technicals['close'])

#Splitting the data to training and test sets
technicals_train = technicals[:round(len(technicals) * 0.80)].copy()
technicals_test  = technicals[round(len(technicals_price) * 0.80):].copy()


#Scaling the features
scaler = MinMaxScaler(feature_range=(-1,1))
technicals_train[['ema_mid','ema_long','ema_short','log_open', 'log_high', 'log_low', 'log_close','atr']] = scaler.fit_transform(technicals_train[['ema_mid','ema_long','ema_short','log_open', 'log_high', 'log_low', 'log_close','atr']])

technicals_test[['ema_mid','ema_long','ema_short','log_open', 'log_high', 'log_low', 'log_close','atr']] = scaler.transform(technicals_test[['ema_mid','ema_long','ema_short','log_open', 'log_high', 'log_low', 'log_close','atr']])

#Creation of input 1 features ( OHLC )
log_prices_test = technicals_test[['log_open', 'log_high', 'log_low', 'log_close']].values.flatten().reshape(len(technicals_test), 4)# 4 represents the OHLC features
log_prices_train = technicals_train[['log_open', 'log_high', 'log_low', 'log_close']].values.flatten().reshape(len(technicals_train), 4)

print("Technicals calculation complete\n\nTime Step calculation beginning now...")

kwargs = {
    'sl': 0.0010, #Desired stop loss ( from entry position)
    'tp':0.0015, #Desired take-profit (from the entry position)
    'look_forward':15, #max length to check if the sl/ tp level has been passed
    'look_back':30, #timestep length desired
    'columns_wanted':['ema_mid', 'ema_long','ema_short','atr']
}

#label and feature creation
train_label , train_indicators = DeepLearning.label_creator(technicals_train, **kwargs)
test_label , test_indicators = DeepLearning.label_creator(technicals_test, **kwargs)

#Obsv distribution of the classes in the train set ( use to determine if class weights should be modified )
print(pd.Series(train_label).value_counts())


#Save the Features input 1
np.save(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/new/test/ohlc_test.npy', log_prices_test)
np.save(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/new/train/ohlc_train.npy', log_prices_train)

#Save the Features input 2
np.save(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/new/test/techs.npy', np.array(test_indicators))
np.save(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/new/train/techs.npy', np.array(train_indicators))

#Save the labels
np.save(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/new/train/label.npy', DeepLearning.label_encode(np.array(train_label)))
np.save(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/new/test/label.npy', DeepLearning.label_encode(np.array(test_label)))

