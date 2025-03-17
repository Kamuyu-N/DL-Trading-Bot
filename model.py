from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Attention
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import  compute_class_weight
from AssistanceFunctions import DeepLearning
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

# Set a maximum memory limit
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=8044)] ) # Limit to 7.9 gigs approx.

#Data ( training )
tp_sl = 'tp_8.7_sl_7.7'

# Input 1
input_1 = Input(shape=(51,4), name='TA')
x1 = LSTM(128, return_sequences=False, name='lstm_TA')(input_1)
x1 = LSTM(64, return_sequences=False, name='LSTM_1_a')(x1)
x1 = LSTM(64,return_sequences=False, name='lstm_1_b')(x1)
x1 = Dropout(0.2)(x1)

#input 2
input_2 = Input(shape=(51,7), name='ASHI')
x2  = LSTM(128,return_sequences=False, name='lstm_ashi')(input_2) # Increase next
x2 = LSTM(64, return_sequences=False, name='LSTM_2_a')(x2)
x2 = Dropout(0.2)(x2)
x2 = LSTM(64,return_sequences=False, name='lstm_2_b')(x2)

merged = Concatenate(name='Merger')([x1,x2])

#Fully connected layers
x = Dense(128, activation='relu', name='dense_1')(x2)
x = Dense(64, activation='relu', name='dense_2')(x1)
x = Dense(32, activation='relu',name= 'dense_3')(x) # try different activations

# Output layer
output = Dense(3, activation='softmax', name='output')(x)

model = Model(inputs=input_1, outputs=output)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),loss=keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.Precision(class_id=0, name='precision_buy'),
        tf.keras.metrics.Recall(class_id=0, name='recall_buy'),
        tf.keras.metrics.Precision(class_id=1, name='precision_sell'),
        tf.keras.metrics.Recall(class_id=1, name='recall_sell')
     ]
)

model.summary()

# implement pruning and regularization techs and fix the metrics to be used
x1_train = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/train/techs.npy', allow_pickle=True)
x2_train = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/train/ashi.npy', allow_pickle=True)
y_train = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/train/label.npy', allow_pickle=True)


print(y_train)
#correct shape of X2
ashi = np.concatenate(x2_train).flatten()
ashi = ashi[~np.array([isinstance(x, pd.Timestamp) for x in ashi])]
x2_train  = ashi.reshape((len(x2_train), 51,7)).astype(np.float32)

#correct shape of X1
tech = np.concatenate(x1_train).flatten()
x1_train = tech.reshape(len(x1_train), 51,4).astype(np.float32)


#Shifting class weights
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
model.fit(x1_train,y_train, batch_size=64, epochs=100 ,verbose=2 ,callbacks=[early_stopping],class_weight=weights_dict,validation_split=0.12)

del x1_train, y_train

#Save the model
model.save("C:/Users/muyu2/OneDrive/Documents/DeepLearning/models/increased_layers_8.7_7.7.h5")

#Data Loading and class weights calc
x1_t = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/test/techs.npy', allow_pickle=True)
x2_t  = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/test/ashi.npy', allow_pickle=True)
y_t = np.load(f'C:/Users/muyu2/OneDrive/Documents/DeepLearning/{tp_sl}/test/label.npy', allow_pickle=True)

ashi = np.concatenate(x2_t).flatten()
ashi = ashi[~np.array([isinstance(x, pd.Timestamp) for x in ashi])]
x2_t  = ashi.reshape((len(x2_t), 51,7)).astype(np.float32)

#correct shape of X1
tech = np.concatenate(x1_t).flatten()
x1_t = tech.reshape(len(x1_t), 51,4).astype(np.float32)

# Split to test and validation set( 60/40 split )
test_len = round(len(x1_t) * 0.6)
x1_test,x2_test , y_test = x1_t[:test_len],x2_t[:test_len], y_t[:test_len]

x1_test, x2_test = DeepLearning.format_test(x1_test, x2_test,y=y_test, timesteps=51,features_x1=4, features_x2=7)
model.evaluate(x1_test, y_test , verbose=2)







