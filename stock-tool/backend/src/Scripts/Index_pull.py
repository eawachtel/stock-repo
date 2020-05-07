import pandas as pd

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

print(stockListCorrected)