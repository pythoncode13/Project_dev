import numpy as np
from datetime import datetime, timedelta


class PositionEvaluator:
    def __init__(
            self,
            df,
            ticker,
            entry_date,
            take_price,
            stop_price,
            position_type,  # 'long' or 'short'
            force_close_minutes,
    ):
        self.df = df
        self.ticker = ticker
        # Convert dateTime to datetime object if it's in string format
        self.current_datetime = datetime.now(tz=self.df['dateTime'].dt.tz)  # Matching timezone with dataframe
        self.take_price = take_price
        self.stop_price = stop_price
        self.position_type = position_type
        self.force_close_date = self._calculate_force_close_date(
            entry_date,
            force_close_minutes
        )
        self.stop_point_index = self._calculate_stop_point_index()
        self.take_point_index = self._calculate_take_point_index()
        self.take_reached, self.take_index = self._is_take_reached()
        self.stop_reached, self.stop_index = self._is_stop_reached()

    def _calculate_force_close_date(self, entry_date, force_close_minutes):
        """Рассчитывает дату принудительного закрытия сделки."""
        date_format = "%Y-%m-%d %H-%M-%S"
        entry_date_pre = datetime.strptime(entry_date, date_format)
        return (
                entry_date_pre.replace(tzinfo=self.df['dateTime'].dt.tz)
                + timedelta(minutes=force_close_minutes)
        )  # Matching timezone

    def _calculate_stop_point_index(self):
        """Ищет индекс свечи, лоу которой достиг стопа."""
        if self.position_type == "long":
            index = self.df.loc[self.df['close'] <= self.stop_price].index.min()
        else:  # short
            index = self.df.loc[self.df['high'] >= self.stop_price].index.min()
        return None if not isinstance(index, (int, np.integer)) else index

    def _calculate_take_point_index(self):
        """Ищет индекс свечи, хай которой достиг тейка."""
        if self.position_type == "long":
            index = self.df.loc[self.df['high'] >= self.take_price].index.min()
        else:  # short
            index = self.df.loc[self.df['low'] <= self.take_price].index.min()
        return None if not isinstance(index, (int, np.integer)) else index

    def evaluate(self):
        """Главная функция проверки достижения уровней."""

        # Если тейк достигнут
        if self.take_reached:
            print(f"{self.ticker} Сделка закрыта по \033[32mтейку\033[0m.")
            return True, "Take", (self.take_index, self.take_price)

        # Если стоп достигнут
        if self.stop_reached:
            print(f"{self.ticker} Сделка закрыта по \033[31mстопу\033[0m.")
            return True, "Stop", (self.stop_index, self.df.loc[self.stop_index, 'close'])

        # Если ни стоп ни тейк не были достигнуты
        # и время принудительного закрытия меньше текущего, закрываем сделку.
        if self.force_close_date <= self.current_datetime:

            closest_index = self.df.index[-1]

            print(
                f"{self.ticker} Сделка закрыта \033[35mпринудительно\033[0m."
            )
            return True, "Force Close", (
                closest_index, self.df.loc[closest_index, 'close']
            )

        return False, None, None

    def _is_take_reached(self):
        """Сравнивает какой из индексов (тейк и стоп) меньше."""
        if self.take_point_index is not None and (
                self.stop_point_index is None
                or self.take_point_index < self.stop_point_index
        ):
            return True, self.take_point_index
        return False, None

    def _is_stop_reached(self):
        """Сравнивает какой из индексов (тейк и стоп) меньше."""
        if self.stop_point_index is not None and (
                self.take_point_index is None
                or self.stop_point_index < self.take_point_index
        ):
            return True, self.stop_point_index
        return False, None
