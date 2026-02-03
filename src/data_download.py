import time
import pandas as pd

from tqdm import tqdm
from tvDatafeed import TvDatafeed, Interval

def get_data_from_tradingview(
        tickers : list,
        interval : Interval,
        exchange : list,
        n_bars : int,
        column : str,
        verbose : bool = True,
        num_trials : int = 5
    ) -> pd.DataFrame:
    """
    Tradingview로부터 가격데이터 import
    """

    data = []

    tv = TvDatafeed() # tradingview 호출
    t = tqdm(zip(tickers, exchange)) if verbose else zip(tickers, exchange)

    for ticker,exchange in t :
        success = False # 성공할때까지 시도
        for attempt in range(num_trials) :
            try :
                temp_data = tv.get_hist(
                    symbol = ticker,
                    exchange = exchange,
                    interval = interval,
                    n_bars = n_bars,
                )
                temp_data.index = pd.to_datetime(
                    temp_data.index.strftime('%Y-%m-%d'),
                )
                data.append(temp_data[column])
                success = True
                break
            except Exception as e :
                if attempt < num_trials - 1 :
                    time.sleep(1)
                else : print(f"Failed to get data from {ticker}. Retrying...")

        if not success : continue

    data = pd.concat(data, axis = 1)
    data.columns = tickers
    return data