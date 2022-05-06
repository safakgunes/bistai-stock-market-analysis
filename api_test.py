import requests

url = "https://stock-market-data.p.rapidapi.com/yfinance/historical-prices"

querystring = {"ticker_symbol":"AKBNK.IS","format":"json","years":"15"}

headers = {
"X-RapidAPI-Host": "stock-market-data.p.rapidapi.com",
"X-RapidAPI-Key": "aa70fdf8f0msh474593534930e2dp105fbdjsnee86c27f3919"
}

response = requests.request("GET", url, headers=headers, params=querystring).json()
print(response)