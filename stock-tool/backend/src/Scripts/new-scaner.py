import yfinance as yf
import csv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from kneed import KneeLocator

class Stock:

    def __init__(self):
        self.basePath = 'C:/stock-repo/ticker-list/'
        self.exportPath = 'C:/watchlist/'
        self.csv = '.csv'

    def getIndexList(self, index, period, interval):
        if index in ['DJI', 'NASDAQ100', 'Futures']:
            path = self.basePath + index + self.csv
            indexStringList = ''
            indexList = []
            with open(path, newline='') as csvfile:
                indexReader = csv.DictReader(csvfile, delimiter=',')
                for row in indexReader:
                    indexStringList = indexStringList + ' ' + row['Symbol']
                    indexList.append(row['Symbol'])
            # testTicker = "TSLA VZ AAPL"
            # testList = ['TSLA', 'VZ', 'AAPL']
        else:
            indexStringList = index
            indexList = [index]

        if interval == '4h':
            interval2 = '1h'
        else:
            interval2 = interval
        indexData = yf.download(tickers=indexStringList,
                                group_by='ticker',
                                period=period,
                                interval=interval2,
                                prepost=True,
                                threads=True)

        if interval == '4h':
            if len(indexList) > 1:
                for stock in indexList:
                    tempStock = indexData[stock]
            if len(indexList) == 1:
                tempstock = indexData

        return indexData, indexList

    def srLevels(self, indexData, indexList, wiggle):
        srWatchList = {}
        for stock in indexList:
            print(stock)
            try:
                if len(indexList) > 1:
                    stockData = indexData[stock]
                if len(indexList) == 1:
                    stockData = indexData
                data = stockData['Close'].tolist()
                close = data[-1]
                closeUp = close + (close * wiggle)
                closeDown = close - (close * wiggle)
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

                Kmean = KMeans(n_clusters=saturation)
                Kmean.fit(X)

                stockElbow = Kmean.cluster_centers_

                majorSR = []
                for i in range(0, saturation):
                    price = stockElbow[i][0]
                    majorSR.append(price)
                    if closeDown < price < closeUp:
                        srWatchList[stock] = {'Price': None}
                        srWatchList[stock]['Price'] = price
                        print(stock + ' added to watchlist')

            except:
                print('Error calculating SR Levels for ' + stock)

        list = []
        for key in srWatchList:
            d = {'ticker': key}
            list.append(d)

        return list

    def export(self, watchList, index, period, interval):
        fileName = index + '_period_' + period + '_interval_' + interval + '.csv'
        watchListDF = pd.DataFrame(watchList)
        export_csv = watchListDF.to_csv(self.exportPath + fileName, index=None,
                                   header=True)


if __name__ == '__main__':
    stocks = Stock()

    ''' Set List to download and option '''
    """Options DJI, NASDAQ100, Futures or enter individual stock as a string value"""
    index = 'TSLA'
    """ valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max"""
    period = "1y"
    """ valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo or 4h for swing """
    interval = "4h"
    """ wiggle is % up/dn from price level"""
    wiggle = .10
    currentIndexData, currentIndexList = stocks.getIndexList(index, period, interval)
    watchList = stocks.srLevels(currentIndexData, currentIndexList, wiggle)
    export = stocks.export(watchList, index, period, interval)

    print('Success')

