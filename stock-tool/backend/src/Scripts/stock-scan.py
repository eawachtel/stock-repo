import pandas as pd
import numpy as np
import sqlite3
import plotly
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from kneed import KneeLocator
import datetime as datetime
import yfinance as yf


class StockData:

    def __init__(self):
        self.db = 'C:/sqlite/stock-data.db'
        self.conn = sqlite3.connect(self.db)
        self.errorList = []

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
            lastInsertedDayDF.columns = col_name_list
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

    def getStockData(self, ticker):
        cur = self.conn.cursor()
        sql = "select * FROM price where Ticker = '{ticker}'".format(ticker=ticker)
        cur.execute(sql)
        col_name_list = [tuple[0] for tuple in cur.description]
        rows = cur.fetchall()
        stockDataDF = pd.DataFrame(rows)
        stockDataDF.columns = col_name_list

        return stockDataDF

    def SRlevels(self, stockData, ticker):

        stockData = stockData
        # stockData = stockData.tail(180)
        close = stockData['High'].tolist()
        open = stockData['Low'].tolist()
        data = close + open
        X = np.array(data).reshape(len(data), 1)
        distortions = []
        inertias = []
        mapping1 = {}
        mapping2 = {}
        K = range(1, 10)

        for k in K:
            # Building and fitting the model
            kmeanModel = KMeans(n_clusters=k).fit(X)
            kmeanModel.fit(X)

            distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                                'euclidean'), axis=1)) / X.shape[0])
            inertias.append(kmeanModel.inertia_)

            mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                           'euclidean'), axis=1)) / X.shape[0]
            mapping2[k] = kmeanModel.inertia_

        distortionsX = range(1, len(distortions) + 1)
        kn = KneeLocator(distortionsX, distortions, S=3, curve='convex', direction='decreasing')
        saturation = kn.knee + 2

        Kmean = KMeans(n_clusters=saturation)
        Kmean.fit(X)

        stockElbow = Kmean.cluster_centers_

        majorSR = []
        for i in range(0, saturation):
            majorSR.append(stockElbow[i][0])

        info = {'ticker': ticker, 'majorSR': majorSR}
        return info

    def ema(self, stockData, ticker, scanResults):
        scanDict = {'ticker': ticker, 'score': 0, 'results': []}
        closingPrice = stockData['Close'].iloc[len(stockData['Close'])-1]
        openPrice = stockData['Close'].iloc[len(stockData['Open']) - 1]
        current200EMA = stockData['EMA200D'].iloc[len(stockData['EMA200D']) - 1]
        current21EMA = stockData['EMA21D'].iloc[len(stockData['EMA21D']) - 1]

        """ Check 200 Day EMA """
        if closingPrice > current200EMA:
            scanDict['results'].append('Above 200D EMA')
            scanDict['score'] = scanDict['score'] + 1
        else:
            scanDict['results'].append('Below 200D EMA')

        """ Check 21 Day EMA """
        if closingPrice > current21EMA:
            scanDict['results'].append('Above 21D EMA')
            scanDict['score'] = scanDict['score'] + 1
        else:
            scanDict['results'].append('Below 21D EMA')
        scanResults.append(scanDict)

        """ Check 21 day pass through """
        if openPrice < current200EMA and closingPrice > current200EMA:
            scanDict['results'].append('200D EMA Pass Through')
            scanDict['score'] = scanDict['score'] + 1

        if openPrice < current21EMA and closingPrice > current21EMA:
            scanDict['results'].append('21D EMA Pass Through')
            scanDict['score'] = scanDict['score'] + 1

        return 'Done'

    def volume(self, stockData, ticker, scanResults, i):

        avgVolume = stockData['Volume'].mean()
        dailyVolume = stockData['Volume'].iloc[len(stockData['Volume']) - 1]
        if dailyVolume > avgVolume:
            scanResults[i]['results'].append('Above avg volume')
            scanResults[i]['score'] = scanResults[i]['score'] + 1

    def bullishEngulf(self, stockData, ticker, scanResults, i):

        """Detect Hammer (wick on the bottom)"""
        close = list(stockData['High'])
        count = 0

        for a in reversed(range(0, len(close))):
            test1 = close[a]
            test2 = close[a - 1]

            if close[a] < close[a - 1]:
                a = a


        return 'Success'


    def RSI(data, period=14):
        # price = StockData(stock)
        series = data
        # print(series)
        delta = series.diff().dropna()
        ups = delta * 0
        downs = ups.copy()
        ups[delta > 0] = delta[delta > 0]
        downs[delta < 0] = -delta[delta < 0]
        ups[ups.index[period - 1]] = np.mean(ups[:period])  # first value is sum of avg gains
        ups = ups.drop(ups.index[:(period - 1)])
        downs[downs.index[period - 1]] = np.mean(downs[:period])  # first value is sum of avg losses
        downs = downs.drop(downs.index[:(period - 1)])
        rs = ups.ewm(com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean() / \
             downs.ewm(com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean()
        return 100 - 100 / (1 + rs)


if __name__ == '__main__':
    StockData = StockData()
    scanResults = []
    dowJones = StockData.getDowJonesStocks()
    for i in range(0, len(dowJones)):
        enter = StockData.downloadStockData(dowJones[i])
        stockData = StockData.getStockData(dowJones[i])
        SRlevels = StockData.SRlevels(stockData, dowJones[i])
        EMA = StockData.ema(stockData, dowJones[i], scanResults)
        volume = StockData.volume(stockData, dowJones[i], scanResults, i)
        # bullishEngulf = StockData.bullishEngulf(stockData, dowJones[i], scanResults, i)
    scanResultsDF = pd.DataFrame(scanResults)
    export_csv = scanResultsDF.to_csv(r'C:\daily\stock-scan.csv', index=None,
                               header=True)
