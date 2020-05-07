import yfinance as yf
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import sqlite3
import datetime as datetime
from get_all_tickers import get_tickers as gt
import threading

class StockData:

    def __init__(self):
        # self.db = 'C:/sqlite/stock-data.db'
        # self.conn = sqlite3.connect(self.db)
        self.errorList = []

    def tickerList(self, dlTickers):

        """Import all NYSE tickers"""
        if dlTickers:
            list = gt.get_tickers()
        return list

    def getDowJonesStocks(self):
        url = r'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
        dfs = pd.read_html(url)
        for df in dfs:
            try:
                if df.columns[2] == 'Symbol':
                    stockList = df['Symbol']
                    stockListCorrected = []
                    for stock in stockList:
                        stockNew = stock.replace('\xa0', '')
                        stockNew2 = stockNew.replace("NYSE:", "")
                        stockListCorrected.append(stockNew2)
                    break

            except:
                print('Not index list')

        return stockListCorrected

    def downloadStockData(self, ticker):
        print('Starting ' + ticker)
        conn = sqlite3.connect('C:/sqlite/stock-data.db')
        try:
            stockObj = yf.Ticker(ticker)
            data = stockObj.history(period="2y", interval='1d')
            # date = data.index
            # date2 = date.strftime('%Y-%m-%d')
            data['Date'] = data.index
            data['Ticker'] = ticker
            data.rename(columns={'Stock Splits': 'Splits'}, inplace=True)
            """Calculate 21d EMA AND 200D ema"""
            data['EMA21D'] = data['Close'].ewm(span=21, adjust=False, min_periods=21).mean()
            data['EMA200D'] = data['Close'].ewm(span=200, adjust=False, min_periods=200).mean()
        except:
            print('Error Connecting to Yahoo API with' + ticker)
            self.errorList.append(ticker)

        """Get current date and only insert new data"""
        now = datetime.datetime.now()

        # try:
        cur = conn.cursor()
        sql = "select max(Date) FROM price where Ticker = '{ticker}'".format(ticker=ticker)
        cur.execute(sql)
        col_name_list = [tuple[0] for tuple in cur.description]
        rows = cur.fetchall()
        lastInsertedDayDF = pd.DataFrame(rows)
        lastDownloadedDay = lastInsertedDayDF.iloc[0, 0]

        """Update if data exist in DB"""
        if lastDownloadedDay is not None:
            lastInsertedDayDF.columns=col_name_list
            data = data[(data.Date > lastDownloadedDay)]
            data.to_sql('price', conn, if_exists='append', index=False)
            print(ticker + ' Data after ' + str(lastDownloadedDay) + ' written to Database')

        """For initial load or for db purge"""
        if lastDownloadedDay is None:
            data.to_sql('price', conn, if_exists='append', index=False)
            print(ticker + ' Full data data period written to Database')


        # except:
        #     print('Error writing ' + ticker + ' to database')

        return 'Success'


if __name__ == '__main__':
    StockData = StockData()
    dlTickers = False
    if dlTickers:
        stockList = StockData.tickerList(True)
    else:
        stockList = ['AMD']
    threads = []
    dowJones = StockData.getDowJonesStocks()
    for ticker in dowJones:
        enter = StockData.downloadStockData(ticker)

    print("Finished updating database")