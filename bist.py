#!/usr/bin/env python
# coding: utf-8

import requests
import sys
from config import API_key
from config import db_pass
import numpy as np
import pandas as pd
import datetime

from ta.volatility import BollingerBands
from os import environ
from sqlalchemy import create_engine
import getpass
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



param_dic = {
     "host"      : "localhost",
    "database"  : "database",
    "user"      : "postgres",
    "password"  : db_pass
}


connect = "postgresql+psycopg2://%s:%s@%s:5432/%s" % (
    param_dic['user'],
    param_dic['password'],
    param_dic['host'],
    param_dic['database']
)
engine = create_engine(connect)

def runAll(symbol):
    response = getStockApi(symbol)
    stock_df = getStockData(response)
    save2DB(symbol, stock_df)
    return getDataFromDB(symbol)

def getStockApi(symbol):

   
    url = "https://stock-market-data.p.rapidapi.com/yfinance/historical-prices"

    querystring = {"ticker_symbol":symbol,"format":"json","years":"15"}

    headers = {
    "X-RapidAPI-Host": "stock-market-data.p.rapidapi.com",
    "X-RapidAPI-Key": "1ce505cd75mshbd09346b7713aa7p150273jsnd2e7a6090e15"
    }

    response = requests.request("GET", url, headers=headers, params=querystring).json()

    return response

def getStockData(response):
    
    year = []
  
    for i in range(10000):
                    try:
                        date = response["historical prices"][i]['Date']
                        year.append(date)
                    except:
                        sys.exit
                        
    trading_days = len(year)

    
    Date = []
    OpenPrice = []
    HighPrice = []
    LowPrice = []
    ClosePrice = []
    VolumeTransactions = []
    adjPrice = []

    for i in range(trading_days):
                        
            try:   
                date = response["historical prices"][i]['Date']
                Date.append(date)
            except:            
                date = np.NAN
                Date.append(date)
        
            try:
                open_price = response["historical prices"][i]['Open']
                OpenPrice.append(open_price)
            except: 
                open_price = np.NAN
                OpenPrice.append(open_price)

            try:  
                high_price = response["historical prices"][i]['High']
                HighPrice.append(high_price)
            except:
                high_price = np.NAN
                HighPrice.append(high_price)
                
            try:
                low_price = response["historical prices"][i]['Low']
                LowPrice.append(low_price)
            except:   
                low_price = np.NAN
                LowPrice.append(low_price)
                
            try:
                close_price = response["historical prices"][i]['Close']
                ClosePrice.append(close_price)  
            except:
                close_price = np.NAN
                ClosePrice.append(close_price)
                
            try:
                volume = response["historical prices"][i]['Volume']
                VolumeTransactions.append(volume)        
            except:
                volume = np.NAN
                VolumeTransactions.append(volume)
                
            try:
                adj_Price = response["historical prices"][i]['Adj Close']
                adjPrice.append(adj_Price)
            except:
                adj_Price = np.NAN
                adjPrice.append(adj_Price)  
            
    #  API data Saved in a Data Frame
      
    DF= pd.DataFrame({'Market_Date': Date,'Open Price ₺': OpenPrice,'High Price ₺':HighPrice,'Low Price ₺': LowPrice,'Close Price ₺': ClosePrice,'Volume':VolumeTransactions,"Adjusted Close Price ₺":adjPrice})

    DF['Market_Date'] = pd.to_datetime(DF['Market_Date'], format='%Y-%m-%d %H:%M:%S.%f').dt.date

    
    stock_df = DF.dropna()
    stock_df.set_index("Market_Date", inplace = True)

    return stock_df


def save2DB(symbol, stock_df):
    stock_df.to_sql('stock_data_'+symbol, con=engine,index=True, if_exists='replace',method='multi')
    print('veriler güncellendi')

def getDataFromDB(symbol):
    result_set = engine.execute('select * from "stock_data_'+symbol+'"')
    df_join = pd.DataFrame(result_set)
    df_join.columns = ['Market_date', 'Open Price ₺', 'High Price ₺','Low Price ₺', 'Close Price ₺', 'Volume','Adjusted Close Price ₺']

    return df_join


#  Machine Learning

def trainModel(stock_df):
    model = LinearRegression()

    X = stock_df[['Open Price ₺','High Price ₺','Low Price ₺','Volume']]
    X.reset_index(drop = True, inplace = True )
    X.to_numpy()

    y = stock_df.pop('Adjusted Close Price ₺')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model.fit(X_train, y_train)

    training_score = model.score(X_train, y_train)
    testing_score = model.score(X_test, y_test)

    print("Training ML is working")
    return model

def predictPrice(model, Open_price, High_price,  Low_price, Volume):
    X_pred = [Open_price,High_price, Low_price, Volume]
    X_pred = np.array(X_pred).reshape(1,4)
    X_pred
    y_pred = model.predict(X_pred)
    return y_pred

def train_and_predict(stock_df, Open_price, High_price,  Low_price, Volume):
    model = trainModel(stock_df)
    return predictPrice(model, Open_price, High_price,  Low_price, Volume)



