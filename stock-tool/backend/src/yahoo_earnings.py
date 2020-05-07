import pandas as pd
import urllib as u
from bs4 import BeautifulSoup as bs
import warnings
warnings.filterwarnings("ignore")
import requests, re, json
p = re.compile(r'root\.App\.main = (.*);')


def get_stock_data(symbol_list):
    try:
        stock_list = []
        for i in symbol_list:
            ticker = i
            analysturl = r'https://finance.yahoo.com/quote/{s}/analysis?p={s}'.format(s=i)
            d = {'ticker':ticker, 'analysturl': analysturl}
            stock_list.append(d)
    except Exception as e:
        print(e)
        pass

    return stock_list


def get_analyst_data(stock_url_list):
    stock_data = []
    try:
        for stock in stock_list:
            analysturl = stock.get('analysturl')
            ticker = stock.get('ticker')
            dfs = pd.read_html(analysturl)
            for df in dfs:
                if df.columns[0] == 'Earnings Estimate':
                    for i in df.index:
                        if df['Earnings Estimate'].iloc[i] == 'Avg. Estimate':
                            avgCurEPS = df.iloc[i, 3]
                        if df['Earnings Estimate'].iloc[i] == 'Avg. Estimate':
                            avgNextEPS = df.iloc[i, 4]
                if df.columns[0] == 'Growth Estimates':
                    for i in df.index:
                        if df['Growth Estimates'].iloc[i] == 'Current Year':
                            CurYrGrowth = df.iloc[i, 1]
                        if df['Growth Estimates'].iloc[i] == 'Next Year':
                            NextYrGrowth = df.iloc[i, 1]
                        if df['Growth Estimates'].iloc[i] == 'Next 5 Years (per annum)':
                            Next5YrGrowthAnual = df.iloc[i, 1]

            with requests.Session() as s:

                r = s.get('https://finance.yahoo.com/quote/{}/key-statistics?p={}'.format(ticker, ticker))
                resdata = json.loads(p.findall(r.text)[0])
                key_stats = resdata['context']['dispatcher']['stores']['QuoteSummaryStore']
                try:
                    Price = key_stats['price']['regularMarketPrice']['fmt']
                except:
                    Price = None
                try:
                    forwardEPS = key_stats['defaultKeyStatistics']['forwardEps']['fmt']
                except:
                    forwardEPS = 0
                try:
                    trailingEPS = key_stats['defaultKeyStatistics']['trailingEps']['fmt']
                except:
                    trailingEPS = 0

            res = {'Price': Price, 'forwardEPS': forwardEPS,'trailingEPS': trailingEPS}

            d = {'ticker': ticker, 'CurYrGrowth': CurYrGrowth, 'NextYrGrowth': NextYrGrowth,
                 '5YrPerAnnumGrowth': Next5YrGrowthAnual, 'avgCurEPS': avgCurEPS, 'avgNextEPS': avgNextEPS,
                 'TrailingEPS': res.get('TrailingEPS'), 'ForwardEPS': res.get('forwardEPS'), 'Price': Price}
            stock_data.append(d)
    except Exception as e:
        print(e)

    return stock_data

def get_growth_calc(analyst_data):
    growth_list = []
    for i in range(0, len(analyst_data)):
        ticker = analyst_data[i].get('ticker')
        avgCurEPS = analyst_data[i].get('avgCurEPS')
        avgNextEPS = analyst_data[i].get('avgNextEPS')
        Price = analyst_data[i].get('Price')
        PEGgrowth = round((float(Price) / avgNextEPS) / (((avgNextEPS - avgCurEPS) / avgCurEPS) * 100), 2)
        d = {'ticker': ticker, 'PEGgrowth': PEGgrowth}
        growth_list.append(d)
    return growth_list

symbol_list = ['AAPL', 'TSLA', 'NKE']
stock_list = get_stock_data(symbol_list)
analyst_data = get_analyst_data(stock_list)
growth_data = get_growth_calc(analyst_data)
test = 'test'