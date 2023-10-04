class PositionEvaluator:
    def __init__(
            self,
            ticker,
            current_datetime,
            take_point_index,
            stop_point_index,
            force_close_date,
    ):

        self.ticker = ticker
        self.current_datetime = current_datetime
        self.take_point_index = take_point_index
        self.stop_point_index = stop_point_index
        self.force_close_date = force_close_date

    def evaluate(self):

        if self._is_take_reached():
            print(f"{self.ticker} Сделка закрыта по \033[32mтейку\033[0m.")
            return True, "Take"

        if self._is_stop_reached():
            print(f"{self.ticker} Сделка закрыта по \033[31mстопу\033[0m.")
            return True, "Stop"

        if self._is_force_closed():
            print(f"{self.ticker} Сделка закрыта \033[35mпринудительно\033[0m.")
            return True, "Force Close"

        return False, None

    def _is_take_reached(self):
        return self.take_point_index is not None and (
            self.stop_point_index is None or self.take_point_index < self.stop_point_index)

    def _is_stop_reached(self):
        return self.stop_point_index is not None and (
            self.take_point_index is None or self.stop_point_index < self.take_point_index)

    def _is_force_closed(self):
        return self.current_datetime >= self.force_close_date
