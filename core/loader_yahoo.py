import yfinance as yf
import pandas as pd
import os
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice

class LoaderYahoo:
    """Загрузчик данных Yahoo Finance."""

    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.df: pd.DataFrame | None = None

    def load_historical_data_online(self):
        """Загрузка истории тикера из Yahoo Finance."""
        stock = yf.Ticker("AAPL")
        self.df = stock.history(period="300d", start=self.start_date,
                                end=self.end_date)

        # Создание столбца dateTime из индекса
        self.df['dateTime'] = self.df.index

        # Переименование столбцов
        self.df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)

        # Если вы хотите, чтобы dateTime был первым столбцом, вы можете изменить порядок столбцов
        self.df = self.df[
            ['open', 'high', 'low', 'close', 'volume', 'dateTime']]


    def get_historical_data(self):
        """Получение истории тикера."""
        self.load_historical_data_online()

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

        return self.df
