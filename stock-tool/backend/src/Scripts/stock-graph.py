import sqlite3
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from kneed import KneeLocator
import plotly.graph_objects as go


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

        return stockDataDF

    def SRlevels(self, stockData):

        stockData = stockData
        stockData = stockData.tail(180)
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
        stockData['5SMA'] = stockData['High'].rolling(window=10).mean()
        # stockData['diff'] = (stockData['High'].diff())

        fig = go.Figure(data=[go.Candlestick(x=stockData.index,
                                             open=stockData['Open'],
                                             high=stockData['High'],
                                             low=stockData['Low'],
                                             close=stockData['Close'],
                                             increasing_line_color='white', decreasing_line_color='red'),
                              go.Scatter(x=stockData.index, y=stockData['EMA21D'], name='21d EMA',
                                         line=dict(color='red', width=4)),
                              go.Scatter(x=stockData.index, y=stockData['EMA200D'], name='200d EMA',
                                         line=dict(color='blue', width=4)),
                              go.Scatter(x=stockData.index, y=stockData['5SMA'], name='5D SMA',
                                         line=dict(color='yellow', width=4))

                              ])

        shapes = []

        for p in SRlevels['majorSR']:
            shape = {'type': 'line',
                     'y0': p,
                     'y1': p,
                     'x0': 0,
                     'x1': len(stockData),
                     'xref': 'x1',
                     'yref': 'y1',
                     'line': {'color': 'green',
                              'width': 4}
                     }
            shapes.append(shape)

        fig.update_layout(
            paper_bgcolor='black',
            plot_bgcolor='black',
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            shapes=shapes
        )
        fig.show()

        return 'graph'


if __name__ == '__main__':
    ticker = 'CAT'
    Graph = Graph()
    stockData = Graph.getStockData(ticker)
    SRlevels = Graph.SRlevels(stockData)
    graph = Graph.graphData(ticker, SRlevels, stockData)
