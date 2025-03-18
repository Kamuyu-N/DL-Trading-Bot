from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import  compute_class_weight
from AssistanceFunctions import DeepLearning
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

# Set a maximum memory limit
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=8044)] ) # Limit to 7.9 gigs approx.

#Data Loading
tp_sl = 'tp_4_sl_9' #Initialize the take profit and stop loss levels used for the feature creation

x1_t = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/test/techs.npy', allow_pickle=True)
x2_t  = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/test/ohlc.npy', allow_pickle=True)
y_t = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/test/label.npy', allow_pickle=True)


# Input 1(Technical indicators)
input_1 = Input(shape=(21, 7), name='TA')
x1 = LSTM(64, return_sequences=False, name='lstm_TA')(input_1)

#input 2 ( Price action - Open , high, Low , close)
input_2 = Input(shape=(21,4), name='OHLC')
x2  = LSTM(128,return_sequences=False, name='lstm_ohlc')(input_2)
x = Dropout(0.2, name='dropout1')(x2)

#concact
merged = Concatenate(name='Merger')([x1,x2])

# Fully connected layers
x = Dense(128, activation='relu', name='dense_1')(merged)
x = Dense(64, activation='relu', name='dense_2')(x)
x = Dense(32, activation='relu',name= 'dense_3')(x) # try different activations

# Output layer
output = Dense(3, activation='softmax', name='output')(x)

model = Model([input_1, input_2], outputs=output)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003),loss=keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.Precision(class_id=0, name='precision_buy'),
        tf.keras.metrics.Recall(class_id=0, name='recall_buy'),
        tf.keras.metrics.Precision(class_id=1, name='precision_sell'),
        tf.keras.metrics.Recall(class_id=1, name='recall_sell')
     ]
)

model.summary()

x1_train = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/train/techs.npy', allow_pickle=True)
x2_train = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/train/ohlc.npy', allow_pickle=True)
y_train = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/train/label.npy', allow_pickle=True)


#correct shape of X2
ohlc = np.concatenate(x2_train).flatten()
x2_train  = ohlc.reshape((len(x2_train), 21, 7))


#Modfication of class weights ( to combat the unbalanced classes )
y_train_int = np.argmax(y_train, axis=1)
classes = np.unique(y_train_int)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_int)

weights_dict = {
    0:weights[0],
    1:weights[1],
    2:weights[2]
}
print(weights_dict)

#class weights to be returned later
model.fit(x1_train,y_train, batch_size=64, epochs=100 ,verbose=2 ,callbacks=[early_stopping],class_weight=weights_dict,validation_split=0.10)

#Data Loading and class weights calc
x1_t = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/test/techs.npy', allow_pickle=True)
x2_t  = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/test/ohlc.npy', allow_pickle=True)
y_t = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/test/label.npy', allow_pickle=True)


#Model evaluation
test_len = round(len(x1_t) * 0.6)# Remaining data is to be used for validation purposes
x1_test,x2_test , y_test = x1_t[:test_len],x2_t[:test_len], y_t[:test_len]

x1_test, x2_test = DeepLearning.format_test(x1_test, x2_test,y=y_test, timesteps=21,features_x1=7, features_x2=4)
model.evaluate([x1_test, x2_test], y_test , verbose=2)

#saving the model
model.save("C:/Users/muyu2/OneDrive/Documents/DeepLearning/models/model_name4.h5")




