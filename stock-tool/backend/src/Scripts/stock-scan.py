import pandas as pd
import numpy as np
import sqlite3
import plotly
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from get_all_tickers import get_tickers as gt
from kneed import KneeLocator
import datetime as datetime
import yfinance as yf
import threading
import time as tm


class StockData:

    def __init__(self):
        self.db = 'C:/sqlite/stock-data.db'
        self.conn = sqlite3.connect(self.db)

    def allTickers(self):
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

    def getYahooData(self, ticker):
        print('Starting ' + ticker)

        try:
            conn = sqlite3.connect('C:/sqlite/stock-data.db')
            stockObj = yf.Ticker(ticker)
            data = stockObj.history(period="2y", interval='1d')
            data['Date'] = data.index
            data['Ticker'] = ticker
            data.rename(columns={'Stock Splits': 'Splits'}, inplace=True)
            """Calculate 21d EMA AND 200D ema"""
            data['EMA8D'] = data['Close'].ewm(span=8, adjust=False, min_periods=8).mean()
            data['EMA21D'] = data['Close'].ewm(span=21, adjust=False, min_periods=21).mean()
            data['EMA50D'] = data['Close'].ewm(span=50, adjust=False, min_periods=50).mean()
            data['EMA200D'] = data['Close'].ewm(span=200, adjust=False, min_periods=200).mean()
            data['RSI'] = self.RSIcalc(data)
            data['PriceDelta'] = data['High'] - data['Low']

        except:
            print('Error Connecting to Yahoo API with' + ticker)


        try:
            """Get current date and only insert new data"""
            now = datetime.datetime.now()

            # try:
            cur = conn.cursor()
            sql = "select max(Date) FROM Daily where Ticker = '{ticker}'".format(ticker=ticker)
            cur.execute(sql)
            col_name_list = [tuple[0] for tuple in cur.description]
            rows = cur.fetchall()
            lastInsertedDayDF = pd.DataFrame(rows)
            lastDownloadedDay = lastInsertedDayDF.iloc[0, 0]

            """Update if data exist in DB"""
            if lastDownloadedDay is not None:
                lastInsertedDayDF.columns = col_name_list
                data = data[(data.Date > lastDownloadedDay)]
                data.to_sql('Daily', conn, if_exists='append', index=False)
                print(ticker + ' Data after ' + str(lastDownloadedDay) + ' written to Database')

            """For initial load or for db purge"""
            if lastDownloadedDay is None:
                data.to_sql('Daily', conn, if_exists='append', index=False)
                print(ticker + ' Full data data period written to Database')

            return 'Success'

        except:
            print('Error writing ' + ticker + ' to database')



    def getStockData(self, ticker):
        try:
            cur = self.conn.cursor()
            sql = "select * FROM Daily where Ticker = '{ticker}'".format(ticker=ticker)
            cur.execute(sql)
            col_name_list = [tuple[0] for tuple in cur.description]
            rows = cur.fetchall()
            stockDataDF = pd.DataFrame(rows)
            stockDataDF.columns = col_name_list

            return stockDataDF

        except:
            print('Error Connecting to DB')



    def SRlevels(self, stockData, ticker, scanResults):
        try:
            stockData = stockData
            # stockData = stockData.tail(180)
            close = stockData['High'].tolist()
            open = stockData['Low'].tolist()
            data = close + open
            cleanedData = [x for x in data if str(x) != 'nan']
            X = np.array(cleanedData).reshape(len(cleanedData), 1)
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

            High = stockData['High'].iloc[len(stockData['High']) - 1]
            Low = stockData['Low'].iloc[len(stockData['Low']) - 1]

            scanResults['MajorSR level'] = 'None'
            scanResults['Within 1 % MajorSR level'] = 'None'
            scanResults['Within 2 % MajorSR level'] = 'None'
            for price in majorSR:
                if Low <= price <= High:
                   scanResults['MajorSR level'] = ('MajorSR level ' + str(price))
                   scanResults['score'] =scanResults['score'] + 1

                elif (Low - (Low * .01)) <= price <= (High + (High * .01)):
                   scanResults['Within 1 % MajorSR level'] = ('Within 1 % MajorSR level ' + str(price))
                   scanResults['score'] =scanResults['score'] + 1

                elif (Low - (Low * .02)) <= price <= (High + (High * .02)):
                   scanResults['Within 2 % MajorSR level'] = ('Within 2 % MajorSR level ' + str(price))
                   scanResults['score'] =scanResults['score'] + 1

            return 'Done'

        except:
            print('Error calculating SR Levels for ' + ticker)
            return []

    def ema(self, stockData, ticker, scanResults):
        try:
            key = ticker
            closingPrice = stockData['Close'].iloc[len(stockData['Close'])-1]
            openPrice = stockData['Open'].iloc[len(stockData['Open']) - 1]
            current200EMA = stockData['EMA200D'].iloc[len(stockData['EMA200D']) - 1]
            current21EMA = stockData['EMA21D'].iloc[len(stockData['EMA21D']) - 1]
            dayAgo21EMA = stockData['EMA21D'].iloc[len(stockData['EMA21D']) - 2]
            current8EMA = stockData['EMA8D'].iloc[len(stockData['EMA8D']) - 1]
            dayAgo8EMA = stockData['EMA8D'].iloc[len(stockData['EMA8D']) - 2]
            EMA8DList = list(stockData['EMA8D'].tail(10))

            if dayAgo8EMA < dayAgo21EMA and current8EMA > current21EMA:
               scanResults['9 day X over 21day'] = '9 day X over 21day'
               scanResults['score'] =scanResults['score'] + 1
            else:
               scanResults['9 day X over 21day'] = 'None'

            if dayAgo8EMA > dayAgo21EMA and (dayAgo21EMA + (dayAgo21EMA * .015)) > dayAgo8EMA and \
                    (EMA8DList[-1] - EMA8DList[-2]) > 0:
               scanResults['T&G EMA touch and up'] = 'T&G EMA touch and up'
               scanResults['score'] = scanResults['score'] + 1
            else:
               scanResults['T&G EMA touch and up'] = 'None'

            # """ Check 200 Day EMA """
            # if closingPrice > current200EMA:
            #    scanResults['results'].append('Above 200D EMA')
            #    scanResults['score'] =scanResults['score'] + 1
            # else:
            #    scanResults['results'].append('Below 200D EMA')
            #
            # """ Check 21 Day EMA """
            # if closingPrice > current21EMA:
            #    scanResults['results'].append('Above 21D EMA')
            #    scanResults['score'] =scanResults['score'] + 1
            # else:
            #    scanResults['results'].append('Below 21D EMA')

            """ Check 21 / 200 day pass through """
            if openPrice < current200EMA and closingPrice > current200EMA:
               scanResults['200D EMA Pass Through'] = '200D EMA Pass Through'
               scanResults['score'] =scanResults['score'] + 1
            else:
               scanResults['200D EMA Pass Through'] = 'None'

            if openPrice < current21EMA and closingPrice > current21EMA:
               scanResults['21D EMA Pass Through'] = '21D EMA Pass Through'
               scanResults['score'] =scanResults['score'] + 1
            else:
               scanResults['21D EMA Pass Through'] = 'None'

            return 'Done'

        except:
            print('EMA Error')





    def RSI(self, stockData, ticker, scanResults):
        try:
            RSI = stockData['RSI'].iloc[len(stockData['RSI']) - 1]
            scanResults['RSI below 50'] = 'None'
            scanResults['RSI below 40'] = 'None'
            scanResults['RSI below 30'] = 'None'
            if 40 <= RSI < 50:
               scanResults['score'] = scanResults['score'] + 1
               scanResults['RSI below 50'] = 'RSI below 50'
            if 30 <= RSI < 40:
               scanResults['score'] = scanResults['score'] + 2
               scanResults['RSI below 40'] = 'RSI below 40'
            if RSI < 30:
               scanResults['score'] = scanResults['score'] + 3
               scanResults['RSI below 30'] = 'RSI below 30'

            return 'Done'

        except:
            print('RSI Error')

    def MACD(self, stockData, ticker, scanResults):
        try:
            stockData['EMA26D'] = stockData['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
            stockData['EMA12D'] = stockData['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
            stockData['MACD'] = stockData["EMA12D"] - stockData["EMA26D"]
            stockData['MACDSig'] = stockData['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()

            MACD = stockData['MACD'].iloc[len(stockData['MACD']) - 1]
            MACDSig = stockData['MACDSig'].iloc[len(stockData['MACDSig']) - 1]

            if MACD > MACDSig:
               scanResults['MACD is bullish'] ='MACD is bullish'
               scanResults['score'] =scanResults['score'] + 1

            return 'MACD'

        except:
            print('MACD Error')

    def volume(self, stockData, ticker, scanResults):
        try:
            avgVolume = stockData['Volume'].mean()
            dailyVolume = stockData['Volume'].iloc[len(stockData['Volume']) - 1]
            if dailyVolume > avgVolume:
               scanResults['Above avg volume'] = 'Above avg volume'
               scanResults ['score'] =scanResults['score'] + 1
            else:
               scanResults['Above avg volume'] = 'None'

            return 'Done'

        except:
            print('Volume Error')

    def bullishEngulf(self, stockData, ticker, scanResults):
        try:
            Close = stockData['Close'].iloc[len(stockData['Close']) - 1]
            Open = stockData['Open'].iloc[len(stockData['Open']) - 1]
            High = stockData['High'].iloc[len(stockData['High']) - 1]
            Low = stockData['Low'].iloc[len(stockData['Low']) - 1]
            delta = stockData['PriceDelta'].iloc[len(stockData['PriceDelta']) - 1]
            dayAgoClose = stockData['Close'].iloc[len(stockData['Close']) - 2]
            dayAgoOpen = stockData['Open'].iloc[len(stockData['Open']) - 2]
            dayAgoHigh = stockData['High'].iloc[len(stockData['High']) - 2]
            dayAgoLow = stockData['Low'].iloc[len(stockData['Low']) - 2]
            dayAgoDelta = stockData['PriceDelta'].iloc[len(stockData['PriceDelta']) - 2]
            twoDayAgoHigh = stockData['High'].iloc[len(stockData['High']) - 3]
            twoDayAgoLow = stockData['Low'].iloc[len(stockData['Low']) - 3]
            twoDayAgoDelta = stockData['PriceDelta'].iloc[len(stockData['PriceDelta']) - 3]
            threeDayAgoHigh = stockData['High'].iloc[len(stockData['High']) - 4]
            threeDayAgoLow = stockData['Low'].iloc[len(stockData['Low']) - 4]

            absDelta = stockData['PriceDelta'].tail(30).abs()
            avgBar = absDelta.mean()

            """Detect bullish engulfing"""
            if dayAgoDelta < 0 and d < dayAgoOpen and Close > dayAgoClose:
               scanResults['Bullish Engulfing'] = 'Bullish Engulfing'
               scanResults['score'] =scanResults['score'] + 2
            else:
               scanResults['Bullish Engulfing'] = 'None'

            """Detect 3 bar play"""
            scanResults['Potential 3 Bar Play Setup'] = 'None'
            if abs(dayAgoDelta) > (avgBar * 2):
                bar2Low = dayAgoLow + (.5 * dayAgoDelta)
                bar2High = dayAgoHigh + (.05 * dayAgoDelta)
                if Low > bar2Low and High < bar2High:
                   scanResults['Potential 3 Bar Play Setup'] = 'Potential 3 Bar Play Setup'
                   scanResults['score'] = scanResults['score'] + 2

            """Detect Inside bar play"""
            scanResults['Bullish 4 Bar Play'] = 'None'
            if abs(twoDayAgoDelta) > (avgBar * 2):
                bar2Low = twoDayAgoLow + (.5 * twoDayAgoDelta)
                bar2High = twoDayAgoHigh + (.05 * twoDayAgoDelta)
                if dayAgoLow > bar2Low and dayAgoHigh < bar2High:
                    if Low > bar2Low and High < bar2High:
                       scanResults['Bullish 4 Bar Play'] = 'Bullish 4 Bar Play'
                       scanResults['score'] =scanResults['score'] + 2

            return 'Success'

        except:
            print('Bullish Error')

    def RSIcalc(self, stockData):
        period = 14
        closingPrice = stockData['Close']
        series = pd.Series(closingPrice)
        delta = series.diff().dropna()
        ups = delta * 0
        downs = ups.copy()
        ups[delta > 0] = delta[delta > 0]
        downs[delta < 0] = -delta[delta < 0]
        ups[ups.index[period - 1]] = np.mean(ups[:period])  # first value is sum of avg gains
        ups = ups.drop(ups.index[:(period - 1)])
        downs[downs.index[period - 1]] = np.mean(downs[:period])  # first value is sum of avg losses
        downs = downs.drop(downs.index[:(period - 1)])
        rs = ups.ewm(com=period - 1,
                     min_periods=0,
                     adjust=False,
                     ignore_na=False).mean() / downs.ewm(com=period - 1,
                       min_periods=0,
                       adjust=False,
                       ignore_na=False).mean()
        rsi = 100 - 100 / (1 + rs)

        return rsi


if __name__ == '__main__':
    StockData = StockData()
    stockList = StockData.allTickers()
    # stockList = StockData.getDowJonesStocks()
    download = True
    scanResultsList = []

    for i in range(0, len(stockList)):
        print(str(i) + ' of ' + str(len(stockList)))
        print('Starting ' + stockList[i] + ' thread')
        scanResults = {}
        scanResults['ticker'] = stockList[i]
        scanResults['score'] = 0
        if download:
            enter = StockData.getYahooData(stockList[i])
        stockData = StockData.getStockData(stockList[i])
        Volume = StockData.volume(stockData, stockList[i], scanResults)
        EMA = StockData.ema(stockData, stockList[i], scanResults)
        RSI = StockData.RSI(stockData, stockList[i], scanResults)
        MACD = StockData.MACD(stockData, stockList[i], scanResults)
        # SRlevels = StockData.SRlevels(stockData, stockList[i], scanResults)
        bullishEngulf = StockData.bullishEngulf(stockData, stockList[i], scanResults)
        scanResultsList.append(scanResults)
        print('Done with ', stockList[i])

    scanResultsDF = pd.DataFrame(scanResultsList)
    export_csv = scanResultsDF.to_csv(r'C:\daily\stock-scan.csv', index=None,
                               header=True)
