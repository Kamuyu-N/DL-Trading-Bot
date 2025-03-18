import pandas as pd
import numpy as np
import pandas_ta as ta
from AssistanceFunctions import DeepLearning
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

pd.options.display.width = 0

#import the data
# price = pd.read_csv('C:/Users/muyu2/OneDrive/Documents/DeepLearning/bbands-test.csv', sep='\t')
price = pd.read_csv('C:/Users/muyu2/OneDrive/Documents/DeepLearning/eurusd-15m.csv', sep=';')

price.columns = ['date', 'time', 'open', 'high','low','close', 'volume']
# price.columns = ['date', 'time', 'open', 'high','low','close','tik_vol', 'volume', 'spreead']
price['datetime'] = price['date'] + " "+ price['time']

# price['datetime'] = pd.to_datetime(price["datetime"], dayfirst=True)
price.drop(['date', 'time'], axis=1, inplace=True)

#Change column arrangement
price = price[['datetime', 'open','high','low','close', 'volume']]

price.drop('volume', axis= 1, inplace=True)

#Create 30 min timeframe data
technicals_price = price.copy()
technicals = price.copy()

# Label creation
x = ta.bbands(price.close, std=2, length=20, mamode='ema')
adx = ta.adx(technicals['high'], technicals['low'], technicals['close'])

technicals = pd.concat([technicals, x, adx], axis=1)


#Addition of other indicators
technicals['ema_mid'] = ta.ema(technicals['close'], length=100)
technicals['ema_long'] = ta.ema(technicals['close'], length=100)
technicals['ema_200'] = ta.ema(technicals['close'], length=200)

technicals.dropna(axis = 0, inplace = True)


technicals['ema_mid'] = (technicals['ema_mid'] - technicals['close'])
technicals['ema_long']= (technicals['ema_long'] - technicals['close'])


technicals_train = technicals[:round(len(technicals) * 0.80)].copy()
technicals_test  = technicals[round(len(technicals_price) * 0.80):].copy()

scaler = MinMaxScaler(feature_range=(-1,1))
technicals_train[['ema_mid','ema_long','DMP_14', 'DMN_14', 'ADX_14']] = scaler.fit_transform(technicals_train[['ema_mid','ema_long','DMP_14', 'DMN_14', 'ADX_14']])

technicals_test[['ema_mid','ema_long','DMP_14', 'DMN_14', 'ADX_14']] = scaler.transform(technicals_test[['ema_mid','ema_long','DMP_14', 'DMN_14', 'ADX_14']])

technicals_test.reset_index(inplace = True, drop = True)
technicals_train.reset_index(inplace = True ,drop = True)

print("Technicals calculation complete\n\nTime Step calculation beginning now...")

kwargs = {
    'forward_length': 16,
    'timestep_length' : 50,
   'indicator_columns': ['ema_long', 'DMP_14', 'DMN_14'],
    'repeat' : False
}

#label creation
train_label , train_indicators = DeepLearning.price_gen_V3(technicals_train, **kwargs)

test_label , test_indicators = DeepLearning.price_gen_V3(technicals_test, **kwargs)

print(pd.Series(train_label).value_counts())#Obsv distribution of the classes in the train set

#Save data
np.save(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/new/test/techs.npy', np.array(test_indicators))
np.save(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/new/train/techs.npy', np.array(train_indicators))

#Save the labels
np.save(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/new/train/label.npy', DeepLearning.label_encode(np.array(train_label)))
np.save(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/new/test/label.npy', DeepLearning.label_encode(np.array(test_label)))

