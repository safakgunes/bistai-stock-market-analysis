# -*- coding: utf-8 -*-

import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import bist
import lstm

st.set_page_config(layout="wide",page_title="bistai - Yapay Zeka Borsa Analizi",page_icon=":shark:")
symbols = tuple(pd.read_csv('stocks.csv')["Symbol"].to_list())
st.sidebar.title("Hisse Senetleri")
symbol = st.sidebar.selectbox(label="Bir Hisse Senedi Seçiniz", options=symbols)
my_page = st.sidebar.radio('Araçlar', ['Site Hakkında','Teknik Analiz','Fiyat Tahmini', 'Anomali Tespiti'])

if symbol:
    df = bist.runAll(symbol)


if my_page == 'Site Hakkında':   

    st.title("BIST30 - Yapay Zeka ile Tahmin ve Anomali Tespiti")
    st.write("Bu Projede , İstanbul Borsasında İşlem Gören En Değerli 30 Hisse Senedinin Geçmişe Yönelik Verilerinin Yapay Zeka Modeli ile İşlenerek ")
    st.write("İstenilen Girdiler ile Yeni Değerlerin Tahmin edilmesi ve Son 2 Yıldaki Fiyatların Anormallik Tespitlerinin Yapılması Amaçlanmıştır.")

    col1,col2 = st.columns([3,1])
    col1.image("stock1.jpg")
    col2.image("kou.png",width=160)
    col2.write("---")
    col2.write("Şafak Güneş")
    col2.write("170208015")
    col2.write("Kocaeli Üniversitesi")
    col2.write("safakcgunes@gmail.com")



elif my_page == 'Teknik Analiz':

    st.title("Keşifsel Veri Analizi")
    st.write("---")

    if symbol:
        st.subheader("Geçmişe Yönelik Fiyat Tablosu")
        st.write(df.sort_values(by="Market_date",ascending=False))
        st.write("---")
        st.subheader("Mum Grafiği")        
        fig = go.Figure(data=[go.Candlestick(x=df['Market_date'], open=df['Open Price ₺'], high=df['High Price ₺'], low=df['Low Price ₺'], close=df['Close Price ₺'])])
        fig.update_layout(width=1400,height=700,yaxis_title="Türk Lirası (₺)")
        st.plotly_chart(fig)






elif  my_page == 'Fiyat Tahmini':

    st.title("Fiyat Tahmini - Lojistik Regresyon")
    st.write("---")

    if symbol:
        df_tahmin = bist.getDataFromDB(symbol)

        with st.form('my_form'):
            input_open = st.number_input("Açılış Fiyatı ₺")
            input_high = st.number_input("En Yüksek Fiyat ₺")
            input_low = st.number_input("En Düşük Fiyat ₺")
            input_vol = st.number_input("Hacim ₺")
            predict_button = st.form_submit_button("Tahmin Et")

            if predict_button:
                prediction = bist.train_and_predict(df_tahmin, input_open, input_high,  input_low, input_vol)
                st.subheader(f'{prediction[0]:.3f}')




else:
    st.title('Anomali Tespiti - LSTM')
    st.write("---")

    if symbol:
       
        test_score_df,anomalies,inverse_test,inverse_anomaly = lstm.lstm(symbol)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_score_df['Market_date'], y=inverse_test, name='Close price'))
        fig.add_trace(go.Scatter(x=anomalies['Market_date'], y=inverse_anomaly, mode='markers', name='Anomaly'))
        fig.update_layout(showlegend=True,width=1400,height=700,yaxis_title="Türk Lirası (₺)")
        st.plotly_chart(fig)
        
       


       


       

   
