import pandas_datareader.data as web
import datetime
import pandas as pd
import numpy as np
import bs4 as bs
import pickle
import requests
import yfinance as yf
import sys
import csv

new_data = input("Do you want to re-download historical data? y/n:  ")

def nyse():
    df = pd.read_csv(r'D:\Stock Data\NYSE.txt', delimiter='\t')
    # print(df)
    # print(df)
    tickers = df.Symbol.tolist()
    tickers2 = [x for x in tickers if "-" not in x]
    tickers3 = [x for x in tickers2 if "." not in x]
    # reader = csv.reader(f)
    # your_list = list(reader)
    return tickers3

def amex():
    # with open('D:\Stock Data\AMEX.txt', 'r') as f:
    df = pd.read_csv(r'D:\Stock Data\AMEX.txt', delimiter='\t')
    # print(df)
    tickers = df.Symbol.tolist()
    tickers2 = [x for x in tickers if "-" not in x]
    tickers3 = [x for x in tickers2 if "." not in x]
        # reader = csv.reader(f)
        # your_list = list(reader)
    return tickers3

def nasdaq():
    # with open('D:\Stock Data\NASDAQ.txt', 'r') as f:
    df = pd.read_csv(r'D:\Stock Data\AMEX.txt', delimiter='\t')
    # print(df)
    tickers = df.Symbol.tolist()
    tickers2 = [x for x in tickers if "-" not in x]
    tickers3 = [x for x in tickers2 if "." not in x]
        # reader = csv.reader(f)
        # your_list = list(reader)
    return tickers3
# nyse()
def sp500():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        tickers.append(ticker)

    # with open("sp500tickers.pickle", "wb") as f:
    #     pickle.dump(tickers, f)
    # tickers = map(lambda s: s.strip(), tickers)
    tickers2 = [x for x in tickers if "-" not in x]
    tickers3 = [x for x in tickers2 if "." not in x]

    return tickers3


# print(save_sp500_tickers())

ewma = pd.Series.ewm
start = datetime.datetime(2019, 8, 1)
end = datetime.datetime(2019, 9, 16)
# stock = 'AAPL'
# price = web.DataReader(stock,'yahoo', start, end)

period = 14

def StockData(stock):
    sdata = web.DataReader(stock, 'yahoo', start, end)
    # LastRSI = RSI('AAPL',price.Close, period)[-1]
    return sdata



def lastPrice(stock, data):
    price = StockData(stock)
    return price.Close[-1]

def RSI(data, period=14):
    # price = StockData(stock)
    series = data
    # print(series)
    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] ) #first value is sum of avg gains
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] ) #first value is sum of avg losses
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() / \
         downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean()
    return 100 - 100 / (1 + rs)

def lastRSI(stock, data):
    return RSI(stock, data)[-1]

# print(RSI('AAPL',price.Close, period)[-1])
# stocks = sp500()
# stocks = nyse3
# stocks = ['AAPL','F','GE','MO','HRB','ULTA','PRU']


def all_stocks():
    list1 = nyse()
    list2 = nasdaq()
    list1 = list1 + list(set(list2) - set(list1))
    list2 = amex()
    list1 = list1 + list(set(list2) - set(list1))
    list2 = sp500()
    list1 = list1 + list(set(list2) - set(list1))

    return list1

# stocks = sp500()
stocks = all_stocks()
# RSI_less40 = []
RSI_dict_less40 = {}
print("working...")
RSIs={}
Screen=[]
dfObj = pd.DataFrame(columns=['Stock','MACD Trend','Close Trend'])

def day50trend(data):
    overall_trend = ''
    i = -50
    while i<-1:
        trend = "-"
        if data[i+1] > data[i]: # -1 > -2
            trend = "+"

        i = i+1
        overall_trend = overall_trend + trend
    return overall_trend

def downloadData():
    print("getting stock data...")
    all_data = yf.download(tickers=" ".join(map(str, stocks)), period='2y', group_by='ticker', threads=True)
    with open(r'D:\Stock Data\all_data.pickle', 'wb') as f:
        pickle.dump(all_data, f)

def getstockData():
    try:
        with open(r'D:\Stock Data\all_data.pickle', 'rb') as f:
            all_data = pickle.load(f)
            print("loading stock data...")
            # print(all_data)
    except:
        downloadData()
    all_data.dropna()
    return all_data

if new_data == 'y':
    downloadData()

all_data = getstockData()
found = 0
print("screening stocks...")
for stock in stocks:
    # print(stock)
    try:
        # print(stock)
        # all_data[stock] = all_data[stock].dropna()
        close_data = all_data[stock]['Adj Close'].dropna()
        volume_data = all_data[stock]['Volume'].dropna()
        last_close = close_data[-1]
        close_max = np.nanmax(close_data)
        close_min = np.nanmin(close_data)
        close_MACD_data = close_data.ewm(span=12).mean() - close_data.ewm(span=26).mean()
        day_50_avg_data = close_data.ewm(span=50).mean()

        # day_50_trend = 'NEG'
        # if day_50_avg_data[-1] > day_50_avg_data[-2]:
        #     day_50_trend = 'POS'

        # print(day_50_trend)
        # day_50_trend_10 = 'NEG'
        # if day_50_avg_data[-9] > day_50_avg_data[-10]:
        #     day_50_trend_10 = 'POS'

        MAC_trend = 'NEG'
        if close_MACD_data[-1] > close_MACD_data[-2]:
            MAC_trend = 'POS'

        close_trend = 'NEG'
        if last_close > close_data[-2]:
            close_trend = 'POS'

        MAC_5_trend = 'NEG'
        if close_MACD_data[-1]>close_MACD_data[-5]:
            MAC_5_trend = 'POS'

        close_5_trend = 'NEG'
        if last_close > close_data[-5]:
            close_5_trend = 'POS'

        MAC_10_trend = 'NEG'
        if close_MACD_data[-1] > close_MACD_data[-10]:
            MAC_10_trend = 'POS'

        close_10_trend = 'NEG'
        if last_close > close_data[-10]:
            close_10_trend = 'POS'

        MAC_20_trend = 'NEG'
        if close_MACD_data[-1] > close_MACD_data[-20]:
            MAC_20_trend = 'POS'

        MAC_50_trend = 'NEG'
        if close_MACD_data[-1] > close_MACD_data[-50]:
            MAC_50_trend = 'POS'

        # print(close_data)

        # if volume_data[-1] > 0 and last_close < close_max * .99 and last_close > close_min * 1.01 and close_MACD_data[-1]>-5 and MAC_5_trend=='POS' and MAC_10_trend == 'POS' and close_5_trend=='POS' and close_10_trend == 'POS':
        if MAC_trend == 'POS' and volume_data[-1]>0 and last_close<50 and last_close>2 and day_50_avg_data[-1]>day_50_avg_data[-2]:
            # print('.', end='')
        # if all_data[stock].Volume[-1] > 0:
        #     print(stock)

            # rsi_data = all_data[stock]['Adj Close']
            RSI_s = RSI(close_data)[-1]
            if RSI_s < 50:
                day_50_trend = day50trend(day_50_avg_data)
                # found = found+1
                # print('.', end='.')
                # print(found, end='\r')
                print(stock, end=' ')
                dfObj = dfObj.append({'Stock' : stock,
                                    'RSI' : RSI_s,
                                    'Adj Close': last_close,
                                    'AdjMax': close_max,
                                    'AdjMin': close_min,
                                    'Volume': volume_data[-1],
                                    'MACD-1': close_MACD_data[-1],
                                    'ABS(MACD)': abs(close_MACD_data[-1]),
                                    'MACD Trend': MAC_trend,
                                    'Close Trend': close_trend,
                                    '50d Avg Trend': day_50_trend,
                                    '50d Avg': day_50_avg_data[-1],
                                  } , ignore_index=True).sort_values(by=['ABS(MACD)'])
    except:
        pass
print('')
pd.options.display.width = 0
print(dfObj)
export_csv = dfObj.to_csv(r'D:\Stock Data\Stock_Screen_9-20-19_BullConv_40RSI_Vol25k.csv', index = None)
tickers = dfObj.Stock.tolist()
print(', '.join(tickers))