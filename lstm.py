from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import pandas as pd
import bist
np.random.seed(1)
tf.random.set_seed(1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

def lstm(symbol) : 
    df = bist.getDataFromDB(symbol)
    df['Market_date'] = pd.to_datetime(df['Market_date'])
    train, test = df.loc[df['Market_date'] <= '2020-01-01'], df.loc[df['Market_date'] > '2020-01-01']
    scaler = StandardScaler()
    scaler = scaler.fit(train[['Close Price ₺']])
    train['Close Price ₺'] = scaler.transform(train[['Close Price ₺']])
    test['Close Price ₺'] = scaler.transform(test[['Close Price ₺']])
    TIME_STEPS=30
    def create_sequences(X, y, time_steps=TIME_STEPS):
        Xs, ys = [], []
        for i in range(len(X)-time_steps):
            Xs.append(X.iloc[i:(i+time_steps)].values)
            ys.append(y.iloc[i+time_steps])
        
        return np.array(Xs), np.array(ys)
    X_train, y_train = create_sequences(train[['Close Price ₺']], train['Close Price ₺'])
    X_test, y_test = create_sequences(test[['Close Price ₺']], test['Close Price ₺'])
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(X_train.shape[1]))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(X_train.shape[2])))
    model.compile(optimizer='adam', loss='mae')
    model.summary()
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)
    model.evaluate(X_test, y_test)
    X_train_pred = model.predict(X_train, verbose=0)
    train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
    threshold = np.max(train_mae_loss)
    X_test_pred = model.predict(X_test, verbose=0)
    test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)
    test_score_df = pd.DataFrame(test[TIME_STEPS:])
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    test_score_df['Close Price ₺'] = test[TIME_STEPS:]['Close Price ₺']
    anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
    inverse_test =scaler.inverse_transform(test_score_df['Close Price ₺'])
    inverse_anomaly= scaler.inverse_transform(anomalies['Close Price ₺'])
    return test_score_df,anomalies,inverse_test,inverse_anomaly

