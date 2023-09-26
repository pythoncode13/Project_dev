import datetime
import os

from binance import Client
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice

import config
from config import (
    api_key,
    api_secret,
)


class LoaderBinance:
    """Загрузчик данных Binance."""

    def __init__(self, ticker, timeframe_interval, s_date, u_date):
        self.client = Client(api_key, api_secret)
        self.df: pd.DataFrame | None = None

        self.ticker = ticker
        self.timeframe_interval = timeframe_interval
        self.s_date = s_date
        self.u_date = u_date

    def get_file_path(self):
        """Получение пути файла."""
        file_name = (
            f'{self.ticker}_{self.timeframe_interval}_'
            f'{self.s_date}_{self.u_date}.xlsx'
        )
        return config.BIGDATA_DIR + file_name

    def load_historical_data_offline(self, historical_path):
        """Загрузка истории тикера из папки bigdata."""
        self.df = pd.read_excel(historical_path)

    def load_historical_data_online(self):
        """Загрузка истории тикера из Бинанса."""
        # since_this_date = datetime.datetime.strptime(self.s_date, '%Y-%m-%d')
        # until_this_date = datetime.datetime.strptime(self.u_date, '%Y-%m-%d')

        since_this_date = datetime.datetime.strptime(self.s_date,
                                                     '%Y-%m-%d %H:%M:%S')
        until_this_date = datetime.datetime.strptime(self.u_date,
                                                     '%Y-%m-%d %H:%M:%S')

        interval = getattr(Client, f'KLINE_INTERVAL_{self.timeframe_interval}')

        # спот, для фьючерсов - self.client.futures_historical_klines
        candle = self.client.get_historical_klines(
            self.ticker,
            interval,
            str(since_this_date),
            str(until_this_date),
        )

        df = pd.DataFrame(
            candle,
            columns=['dateTime', 'open', 'high', 'low', 'close',
                     'volume', 'closeTime',
                     'quoteAssetVolume', 'numberOfTrades',
                     'takerBuyBaseVol',
                     'takerBuyQuoteVol', 'ignore'],
        )
        df.dateTime = pd.to_datetime(df.dateTime, unit='ms').dt.strftime(
            "%Y-%m-%d %H-%M-%S")

        date_index = df['dateTime']

        df = df.drop(
            ['dateTime', 'closeTime', 'quoteAssetVolume', 'numberOfTrades',
             'takerBuyQuoteVol', 'ignore'],
            axis=1,
        )

        df = df.apply(pd.to_numeric)
        df['dateTime'] = date_index

        self.df = df

    def add_indicators(self) -> None:
        """Добавляем индикаторы в базу."""
        # RSI
        rsi = RSIIndicator(self.df['close'], window=14)
        self.df['rsi'] = rsi.rsi()

        # VWAP
        vwap = VolumeWeightedAveragePrice(
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            volume=self.df['volume'],
        )
        self.df['vwap2'] = vwap.vwap
        self.df['vwap'] = (
            (
                self.df['volume'] *
                (self.df['high'] + self.df['low'] + self.df[
                    'close']) / 3
            ).cumsum() / self.df['volume'].cumsum()
        )

        self.df = self.df.iloc[14:]
        self.df = self.df.reset_index(drop=True)

    def get_historical_data(self):
        """Получение истории тикера."""
        # historical_path = self.get_file_path()
        # if os.path.exists(historical_path):
        #     self.load_historical_data_offline(historical_path)
        # else:
        self.load_historical_data_online()
        # self.df.to_excel(historical_path)

        return self.df
