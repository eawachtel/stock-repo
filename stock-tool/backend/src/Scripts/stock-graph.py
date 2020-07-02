import sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from kneed import KneeLocator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Graph:

    def __init__(self):
        self.db = 'C:/sqlite/stock-data.db'
        self.conn = sqlite3.connect(self.db)
        self.errorList = []

    def getStockData(self, ticker):
        cur = self.conn.cursor()
        sql = "select * FROM price where Ticker = '{ticker}'".format(ticker=ticker)
        cur.execute(sql)
        col_name_list = [tuple[0] for tuple in cur.description]
        rows = cur.fetchall()
        stockDataDF = pd.DataFrame(rows)
        stockDataDF.columns = col_name_list
        stockDataDF['RSI'] = self.RSIcalc(stockDataDF)
        stockDataDF = self.MACDcalc(stockDataDF)

        return stockDataDF

    def MACDcalc(self, stockData):

        stockData['EMA26D'] = stockData['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
        stockData['EMA12D'] = stockData['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
        stockData['MACD'] = stockData["EMA12D"] - stockData["EMA26D"]
        stockData['MACDSig'] = stockData['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()

        return stockData

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
        rs = ups.ewm(com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean() / \
             downs.ewm(com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean()
        return 100 - 100 / (1 + rs)

    def SRlevels(self, stockData):

        stockData = stockData
        stockData = stockData.tail(180)
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
        saturation = kn.knee
        # saturation = 10
        Kmean = KMeans(n_clusters=saturation)
        Kmean.fit(X)

        stockElbow = Kmean.cluster_centers_

        majorSR = []
        for i in range(0, saturation):
            majorSR.append(stockElbow[i][0])

        info = {'ticker': ticker, 'majorSR': majorSR}
        return info

    def graphData(self, Ticker, SRlevels, stockData):
        stockData['5SMA'] = stockData['High'].rolling(window=5).mean()
        # stockData['diff'] = (stockData['High'].diff())

        fig = make_subplots(rows=4, cols=1, row_heights=[500, 0, 100, 100])

        fig.add_trace(go.Candlestick(x=stockData.index,
                                             open=stockData['Open'],
                                             high=stockData['High'],
                                             low=stockData['Low'],
                                             close=stockData['Close'],
                                             increasing_line_color='white', decreasing_line_color='red'),

                      row=1, col=1)

        # fig.add_trace(go.Scatter(x=stockData.index, y=stockData['EMA21D'], name='21d EMA',
        #          line=dict(color='red', width=4)),
        #             row=1, col=1
        #             )
        # fig.add_trace(go.Scatter(x=stockData.index, y=stockData['EMA200D'], name='200d EMA',
        #                          line=dict(color='blue', width=4)),row=1, col=1)

        fig.add_trace(go.Scatter(x=stockData.index, y=stockData['5SMA'], name='5D SMA',
                                 line=dict(color='yellow', width=4)),row=1, col=1)

        fig.add_trace(go.Scatter(x=stockData.index, y=stockData['RSI'], name='RSI',
                                 line=dict(color='white', width=4)), row=3, col=1)

        fig.add_trace(go.Scatter(x=stockData.index, y=stockData['MACD'], name='MACD',
                                 line=dict(color='white', width=4)), row=4, col=1)

        fig.add_trace(go.Scatter(x=stockData.index, y=stockData['MACDSig'], name='MACDEXP',
                                 line=dict(color='blue', width=4)), row=4, col=1)

        fig.add_shape(
            # Line Horizontal
            type="line",
            x0=0,
            y0=30,
            x1=len(stockData.index),
            y1=30,
            line=dict(
                color="red",
                width=4
            ),  row=3, col=1
        )

        fig.add_shape(
            # Line Horizontal
            type="line",
            x0=0,
            y0=60,
            x1=len(stockData.index),
            y1=60,
            line=dict(
                color="red",
                width=4
            ), row=3, col=1
        )

        shapes = []

        for p in SRlevels['majorSR']:
            fig.add_shape(type="line",
                x0=0,
                y0=p,
                x1=len(stockData.index),
                y1=p,
                line=dict(
                    color="red",
                    width=4
                ), row=1, col=1
            )

        fig.update_layout(
            xaxis_rangeslider_visible=False,
            paper_bgcolor='black',
            plot_bgcolor='black',
            xaxis_showgrid=False,
            yaxis_showgrid=False,

        )
        fig.show()

        return 'graph'


if __name__ == '__main__':
    ticker = 'MMM'
    Graph = Graph()
    stockData = Graph.getStockData(ticker)
    SRlevels = Graph.SRlevels(stockData)
    graph = Graph.graphData(ticker, SRlevels, stockData)
