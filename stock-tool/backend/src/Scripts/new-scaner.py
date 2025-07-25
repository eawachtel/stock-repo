import yfinance as yf
import csv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from kneed import KneeLocator
import time
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Stock:

    def __init__(self):
        self.basePath = 'index_files/'
        self.csv = '.csv'

    def getIndexList(self, index, period, interval):
        """
        Download stock data with improved error handling and 4h interval support
        """
        # Get ticker list
        if index in ['DJI', 'NASDAQ100', 'Futures']:
            path = self.basePath + index + self.csv
            indexStringList = ''
            indexList = []
            try:
                with open(path, newline='') as csvfile:
                    indexReader = csv.DictReader(csvfile, delimiter=',')
                    for row in indexReader:
                        indexStringList = indexStringList + ' ' + row['Symbol']
                        indexList.append(row['Symbol'])
                logger.info(f"Loaded {len(indexList)} tickers from {index}")
            except FileNotFoundError:
                logger.error(f"File not found: {path}")
                return None, []
        else:
            indexStringList = index
            indexList = [index]

        # Handle 4h interval properly
        download_interval = '1h' if interval == '4h' else interval
        
        # Download data with retries and error handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}: Downloading data for {len(indexList)} ticker(s)")
                logger.info(f"Parameters: period={period}, interval={download_interval}")
                
                indexData = yf.download(
                    tickers=indexStringList,
                    period=period,
                    interval=download_interval
                )
                
                # Check if data was downloaded successfully
                if indexData is None or indexData.empty:
                    logger.warning(f"No data returned for attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                        continue
                    else:
                        logger.error("Failed to download data after all attempts")
                        return None, indexList
                
                # Handle 4h interval conversion
                if interval == '4h':
                    indexData = self.convert_to_4h(indexData, indexList)
                
                logger.info("Data downloaded successfully")
                return indexData, indexList
                
            except Exception as e:
                logger.error(f"Error downloading data (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait longer before retry
                    continue
                else:
                    logger.error("Failed to download data after all attempts")
                    return None, indexList

    def convert_to_4h(self, data, ticker_list):
        """
        Convert 1h data to 4h data - handles MultiIndex structure (TICKER, PRICE_TYPE)
        """
        try:
            logger.info(f"Converting to 4h data. Data shape: {data.shape}")
            logger.info(f"Data columns: {data.columns.tolist()}")
            
            if len(ticker_list) == 1:
                ticker = ticker_list[0]
                
                # Your yfinance version always returns MultiIndex: (TICKER, PRICE_TYPE)
                if isinstance(data.columns, pd.MultiIndex):
                    # Find columns for this ticker: (TICKER, PRICE_TYPE) format
                    ticker_columns = {}
                    
                    for col_tuple in data.columns:
                        if len(col_tuple) == 2 and col_tuple[0] == ticker:
                            price_type = col_tuple[1]  # 'Open', 'High', 'Low', 'Close', 'Volume'
                            ticker_columns[price_type] = col_tuple
                    
                    logger.info(f"Found ticker columns: {ticker_columns}")
                    
                    if not ticker_columns:
                        logger.warning("No ticker columns found, returning original data")
                        return data
                    
                    # Create aggregation dictionary using the full tuple names
                    agg_dict = {}
                    for price_type, full_col in ticker_columns.items():
                        if price_type == 'Open':
                            agg_dict[full_col] = 'first'
                        elif price_type == 'High':
                            agg_dict[full_col] = 'max'
                        elif price_type == 'Low':
                            agg_dict[full_col] = 'min'
                        elif price_type == 'Close':
                            agg_dict[full_col] = 'last'
                        elif price_type == 'Volume':
                            agg_dict[full_col] = 'sum'
                    
                    logger.info(f"Aggregation dictionary: {agg_dict}")
                    
                    if not agg_dict:
                        logger.warning("No aggregation columns found, returning original data")
                        return data
                    
                    resampled = data.resample('4h').agg(agg_dict).dropna()
                    
                    # Keep the MultiIndex structure but log it
                    logger.info(f"Resampled data shape: {resampled.shape}")
                    logger.info(f"Final columns: {resampled.columns.tolist()}")
                    return resampled
                else:
                    # Fallback for regular DataFrame structure
                    agg_dict = {
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }
                    # Only include columns that exist
                    agg_dict = {k: v for k, v in agg_dict.items() if k in data.columns}
                    
                    resampled = data.resample('4h').agg(agg_dict).dropna()
                    return resampled
            else:
                # Multiple tickers - should also be MultiIndex (TICKER, PRICE_TYPE)
                logger.info("Processing multiple tickers")
                # For multiple tickers, the structure should already be correct
                # Just resample each ticker's data
                
                if isinstance(data.columns, pd.MultiIndex):
                    # Group by ticker and resample each
                    resampled_data = {}
                    
                    for ticker in ticker_list:
                        # Extract columns for this ticker
                        ticker_cols = [col for col in data.columns if col[0] == ticker]
                        
                        if ticker_cols:
                            ticker_data = data[ticker_cols]
                            
                            # Create aggregation dict with full column names
                            agg_dict = {}
                            for col in ticker_cols:
                                price_type = col[1]
                                if price_type == 'Open':
                                    agg_dict[col] = 'first'
                                elif price_type == 'High':
                                    agg_dict[col] = 'max'
                                elif price_type == 'Low':
                                    agg_dict[col] = 'min'
                                elif price_type == 'Close':
                                    agg_dict[col] = 'last'
                                elif price_type == 'Volume':
                                    agg_dict[col] = 'sum'
                            
                            resampled_ticker = ticker_data.resample('4h').agg(agg_dict).dropna()
                            resampled_data[ticker] = resampled_ticker
                    
                    # Combine all tickers
                    if resampled_data:
                        combined = pd.concat(resampled_data.values(), axis=1)
                        return combined
                
                return data
                
        except Exception as e:
            logger.error(f"Error converting to 4h data: {str(e)}")
            logger.info("Returning original data without resampling")
            return data

    def validate_stock_data(self, stock_data, ticker):
        """
        Validate that stock data is usable - handles both MultiIndex structures
        """
        if stock_data is None or stock_data.empty:
            logger.warning(f"No data for {ticker}")
            return False
        
        logger.info(f"Validating {ticker}. Columns: {stock_data.columns.tolist()}")
        
        # Find Close column - handle both MultiIndex structures
        close_col = None
        if isinstance(stock_data.columns, pd.MultiIndex):
            # Look for both patterns: ('TSLA', 'Close') and ('Close', 'TSLA')
            for col_tuple in stock_data.columns:
                if len(col_tuple) == 2:
                    if col_tuple[0] == ticker and col_tuple[1] == 'Close':  # ('TSLA', 'Close')
                        close_col = col_tuple
                        break
                    elif col_tuple[1] == ticker and col_tuple[0] == 'Close':  # ('Close', 'TSLA')
                        close_col = col_tuple
                        break
        else:
            # Regular columns
            for col in stock_data.columns:
                if 'Close' in str(col):
                    close_col = col
                    break
        
        if close_col is None:
            logger.warning(f"No Close column for {ticker}")
            return False
        
        close_data = stock_data[close_col].dropna()
        if len(close_data) < 10:  # Need minimum data points
            logger.warning(f"Insufficient data points for {ticker}: {len(close_data)}")
            return False
        
        logger.info(f"Validation passed for {ticker}. Close column: {close_col}")
        return True

    def srLevels(self, indexData, indexList, wiggle):
        """
        Calculate support/resistance levels with improved error handling
        """
        if indexData is None:
            logger.error("No data available for SR calculation")
            return []
        
        srWatchList = {}
        
        for stock in indexList:
            logger.info(f"Processing {stock}")
            
            try:
                # Get stock data - handle both MultiIndex structures
                if len(indexList) > 1:
                    # For multiple tickers, extract this ticker's data from MultiIndex
                    stockData = pd.DataFrame()
                    for col_tuple in indexData.columns:
                        if len(col_tuple) == 2:
                            if col_tuple[0] == stock:  # ('TSLA', 'Close')
                                col_type = col_tuple[1]
                                stockData[col_type] = indexData[col_tuple]
                            elif col_tuple[1] == stock:  # ('Close', 'TSLA')
                                col_type = col_tuple[0]
                                stockData[col_type] = indexData[col_tuple]
                else:
                    # Single ticker
                    stockData = indexData
                
                # Validate data
                if not self.validate_stock_data(stockData, stock):
                    continue
                
                # Process close prices - handle both MultiIndex structures
                close_col = None
                if isinstance(stockData.columns, pd.MultiIndex):
                    # Look for both patterns: ('TSLA', 'Close') and ('Close', 'TSLA')
                    for col_tuple in stockData.columns:
                        if len(col_tuple) == 2:
                            if col_tuple[0] == stock and col_tuple[1] == 'Close':  # ('TSLA', 'Close')
                                close_col = col_tuple
                                break
                            elif col_tuple[1] == stock and col_tuple[0] == 'Close':  # ('Close', 'TSLA')
                                close_col = col_tuple
                                break
                else:
                    # Regular columns or already processed
                    for col in stockData.columns:
                        if 'Close' in str(col):
                            close_col = col
                            break
                
                if close_col is None:
                    logger.warning(f"No Close column found for {stock}")
                    continue
                
                data = stockData[close_col].dropna().tolist()
                if len(data) < 10:
                    logger.warning(f"Insufficient data for {stock}")
                    continue
                
                close = data[-1]
                closeUp = close + (close * wiggle)
                closeDown = close - (close * wiggle)
                
                # Prepare data for clustering
                X = np.array(data).reshape(len(data), 1)
                
                # Find optimal number of clusters
                distortions = []
                K = range(1, min(10, len(data) // 2))  # Ensure we don't exceed data points
                
                for k in K:
                    try:
                        kmeanModel = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeanModel.fit(X)
                        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                                            'euclidean'), axis=1)) / X.shape[0])
                    except Exception as e:
                        logger.warning(f"Error in clustering for k={k}: {str(e)}")
                        continue
                
                if not distortions:
                    logger.warning(f"No valid clusters found for {stock}")
                    continue
                
                # Find elbow point
                try:
                    distortionsX = range(1, len(distortions) + 1)
                    kn = KneeLocator(distortionsX, distortions, S=3, curve='convex', direction='decreasing')
                    saturation = kn.knee if kn.knee else min(3, len(distortions))
                except Exception as e:
                    logger.warning(f"Error finding elbow for {stock}: {str(e)}")
                    saturation = min(3, len(distortions))
                
                # Apply K-means with optimal clusters
                try:
                    Kmean = KMeans(n_clusters=saturation, random_state=42, n_init=10)
                    Kmean.fit(X)
                    stockElbow = Kmean.cluster_centers_
                    
                    # Store all SR levels for this stock (not just those near current price)
                    all_levels = [level[0] for level in stockElbow]
                    
                    # Check for levels near current price
                    for i in range(saturation):
                        price = stockElbow[i][0]
                        if closeDown < price < closeUp:
                            srWatchList[stock] = {
                                'Price': price, 
                                'Current': close,
                                'All_Levels': all_levels  # Store all levels for plotting
                            }
                            logger.info(f'{stock} added to watchlist at level {price:.2f}')
                            break
                    
                    # If no levels near current price, still store for potential plotting
                    if stock not in srWatchList:
                        srWatchList[stock] = {
                            'Price': None,
                            'Current': close,
                            'All_Levels': all_levels
                        }
                            
                except Exception as e:
                    logger.error(f"Error in final clustering for {stock}: {str(e)}")
                    continue

            except Exception as e:
                logger.error(f'Error calculating SR Levels for {stock}: {str(e)}')
                continue

        # Convert to list format
        watchlist = []
        for ticker, data in srWatchList.items():
            watchlist.append({
                'ticker': ticker,
                'sr_level': data['Price'],
                'current_price': data['Current'],
                'all_levels': data['All_Levels']
            })

        logger.info(f"Found {len(watchlist)} stocks with SR analysis")
        return watchlist

    def get_ohlc_data(self, indexData, ticker):
        """
        Extract OHLC data for a specific ticker from the downloaded data
        """
        try:
            logger.info(f"Extracting OHLC data for {ticker}")
            logger.info(f"Available columns: {indexData.columns.tolist()}")
            
            ohlc_data = pd.DataFrame()
            
            if isinstance(indexData.columns, pd.MultiIndex):
                # Handle MultiIndex structure - look for both patterns
                ticker_columns = {}
                
                for col_tuple in indexData.columns:
                    if len(col_tuple) == 2:
                        # Pattern 1: (TICKER, PRICE_TYPE) like ('TSLA', 'Close')
                        if col_tuple[0] == ticker:
                            price_type = col_tuple[1]
                            ticker_columns[price_type] = col_tuple
                            logger.info(f"Found column: {col_tuple} -> {price_type}")
                        # Pattern 2: (PRICE_TYPE, TICKER) like ('Close', 'TSLA')
                        elif col_tuple[1] == ticker:
                            price_type = col_tuple[0]
                            ticker_columns[price_type] = col_tuple
                            logger.info(f"Found column: {col_tuple} -> {price_type}")
                
                logger.info(f"Ticker columns found: {ticker_columns}")
                
                # Extract data for each price type
                for price_type, full_col in ticker_columns.items():
                    ohlc_data[price_type] = indexData[full_col]
                
            else:
                # Handle regular columns (single ticker case)
                logger.info("Using regular column structure")
                ohlc_data = indexData.copy()
            
            logger.info(f"OHLC data columns after extraction: {ohlc_data.columns.tolist()}")
            
            # Check for required columns with flexible matching
            required_cols = ['Open', 'High', 'Low', 'Close']
            available_cols = ohlc_data.columns.tolist()
            
            # Create mapping for case-insensitive and flexible matching
            col_mapping = {}
            for req_col in required_cols:
                found = False
                for avail_col in available_cols:
                    if req_col.lower() in str(avail_col).lower():
                        col_mapping[req_col] = avail_col
                        found = True
                        break
                if not found:
                    logger.error(f"Missing required column: {req_col}")
                    logger.error(f"Available columns: {available_cols}")
                    return None
            
            # Rename columns to standard names if needed
            if col_mapping != {col: col for col in required_cols}:
                logger.info(f"Column mapping: {col_mapping}")
                ohlc_data = ohlc_data.rename(columns={v: k for k, v in col_mapping.items()})
            
            # Ensure we have the required columns
            if not all(col in ohlc_data.columns for col in required_cols):
                logger.error(f"Missing required OHLC columns for {ticker}")
                logger.error(f"Required: {required_cols}")
                logger.error(f"Available: {ohlc_data.columns.tolist()}")
                return None
            
            # Remove any rows with NaN values
            ohlc_data = ohlc_data.dropna()
            
            if ohlc_data.empty:
                logger.error(f"No valid OHLC data for {ticker}")
                return None
            
            logger.info(f"Successfully extracted OHLC data for {ticker}. Shape: {ohlc_data.shape}")
            return ohlc_data
            
        except Exception as e:
            logger.error(f"Error extracting OHLC data for {ticker}: {str(e)}")
            return None

    def test_plot(self, indexData, watchlist, ticker=None):
        """
        Plot candlestick chart with support/resistance levels
        """
        try:
            # If ticker is not specified, use the first ticker from watchlist
            if ticker is None:
                if not watchlist:
                    logger.error("No watchlist data available for plotting")
                    return
                ticker = watchlist[0]['ticker']
            
            # Find the watchlist entry for this ticker
            ticker_data = None
            for item in watchlist:
                if item['ticker'] == ticker:
                    ticker_data = item
                    break
            
            if ticker_data is None:
                logger.error(f"No watchlist data found for {ticker}")
                return
            
            # Get OHLC data
            ohlc_data = self.get_ohlc_data(indexData, ticker)
            if ohlc_data is None:
                return
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot candlesticks
            for i, (date, row) in enumerate(ohlc_data.iterrows()):
                open_price = row['Open']
                high_price = row['High']
                low_price = row['Low']
                close_price = row['Close']
                
                # Determine color (green for up, red for down)
                color = 'green' if close_price >= open_price else 'red'
                
                # Draw the high-low line
                ax.plot([i, i], [low_price, high_price], color='black', linewidth=1)
                
                # Draw the body
                body_height = abs(close_price - open_price)
                body_bottom = min(open_price, close_price)
                
                rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                               facecolor=color, alpha=0.7, edgecolor='black')
                ax.add_patch(rect)
            
            # Plot support/resistance levels as horizontal lines
            if ticker_data['all_levels']:
                for level in ticker_data['all_levels']:
                    ax.axhline(y=level, color='black', linestyle='-', linewidth=2, alpha=0.8)
                    # Add level label
                    ax.text(len(ohlc_data) * 1.01, level, f'${level:.2f}', 
                           verticalalignment='center', fontsize=10, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # Highlight the current watchlist level if it exists
            if ticker_data['sr_level'] is not None:
                ax.axhline(y=ticker_data['sr_level'], color='blue', linestyle='--', 
                          linewidth=3, alpha=0.9, label=f'Watchlist Level: ${ticker_data["sr_level"]:.2f}')
            
            # Mark current price
            current_price = ticker_data['current_price']
            ax.axhline(y=current_price, color='orange', linestyle=':', 
                      linewidth=2, alpha=0.9, label=f'Current Price: ${current_price:.2f}')
            
            # Formatting
            ax.set_title(f'{ticker} - Candlestick Chart with Support/Resistance Levels', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Time Period', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format x-axis with dates if possible
            if len(ohlc_data) > 0:
                # Create date labels for every 10th candle to avoid crowding
                step = max(1, len(ohlc_data) // 10)
                date_indices = range(0, len(ohlc_data), step)
                date_labels = [ohlc_data.index[i].strftime('%m/%d %H:%M') if hasattr(ohlc_data.index[i], 'strftime') 
                              else str(i) for i in date_indices]
                ax.set_xticks(date_indices)
                ax.set_xticklabels(date_labels, rotation=45)
            
            plt.tight_layout()
            plt.show()
            
            logger.info(f"Successfully plotted candlestick chart for {ticker}")
            
        except Exception as e:
            logger.error(f"Error creating plot for {ticker}: {str(e)}")

    def export(self, watchList, index, period, interval):
        """
        Export watchlist with error handling
        """
        try:
            fileName = f"{index}_period_{period}_interval_{interval}.csv"
            
            # Prepare data for export (exclude all_levels as it's not needed in CSV)
            export_data = []
            for item in watchList:
                export_data.append({
                    'ticker': item['ticker'],
                    'sr_level': item['sr_level'],
                    'current_price': item['current_price']
                })
            
            watchListDF = pd.DataFrame(export_data)
            
            if watchListDF.empty:
                logger.warning("No data to export")
                return
            
            export_path = fileName
            watchListDF.to_csv(export_path, index=False, header=True)
            logger.info(f"Exported {len(export_data)} items to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")


if __name__ == '__main__':
    stocks = Stock()

    # Configuration
    index = 'DJI'  # Options: DJI, NASDAQ100, SP500, Futures, or individual stock
    period = "1y"   # Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    interval = "1d" # Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo or 4h
    wiggle = 0.10   # % up/down from price level

    try:
        logger.info("Starting stock scanner...")
        
        # Download data
        currentIndexData, currentIndexList = stocks.getIndexList(index, period, interval)
        
        if currentIndexData is None:
            logger.error("Failed to download data. Exiting.")
            exit(1)
        
        # Calculate SR levels
        watchList = stocks.srLevels(currentIndexData, currentIndexList, wiggle)
        
        # Export results
        stocks.export(watchList, index, period, interval)
        
        # Create plot
        if watchList:
            stocks.test_plot(currentIndexData, watchList, index)
        else:
            logger.info("No data to plot")
        
        logger.info('Process completed successfully')
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        exit(1)