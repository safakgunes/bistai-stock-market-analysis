import requests

url = "https://yahoofinance-stocks1.p.rapidapi.com/stock-prices"

querystring = {"EndDateInclusive":"2020-04-01","StartDateInclusive":"2002-01-01","Symbol":"MSFT","OrderBy":"Ascending"}

headers = {
	"X-RapidAPI-Host": "yahoofinance-stocks1.p.rapidapi.com",
	"X-RapidAPI-Key": "0bc6d85258msh0169da777118572p18eaf1jsnfdb23e2857d9"
}

response = requests.request("GET", url, headers=headers, params=querystring).json()

print(response)