class BaseTradeAnalysis:
    """Класс, в котором происходит расчет базовых параметров сделки."""
    def __init__(self, t1up, t4up, t3up, entry_price, stop_price, take_price):
        self.t1up = t1up
        self.t2up = t4up
        self.t3up = t3up

        self.entry_price = entry_price
        self.stop_price = stop_price
        self.take_price = take_price

    @property
    def calculate_risk_reward_ratio(self):
        """Считает соотношение риск/прибыль."""

        potential_risk = self.entry_price - self.stop_price
        potential_reward = self.take_price - self.entry_price

        if potential_risk != 0:  # Избегаем деления на ноль
            risk_reward_ratio = potential_reward / potential_risk
        else:
            risk_reward_ratio = float('inf')  # Бесконечность

        return risk_reward_ratio

    @property
    def percent_difference(self):
        return abs(
            ((self.entry_price - self.take_price) / self.entry_price) * 100
        )

    @property
    def stop_percent_difference(self):
        return 100 - self.stop_price * 100 / self.entry_price

    @property
    def take_percent_difference(self):
        return self.take_price * 100 / self.entry_price - 100

    # @property
    # def entry_price(self):
    #     """Назначает цену входа."""
    #     self._entry_price = self.t2up[1]
    #
    #     return self._entry_price

    # @property
    # def up_take_lines(self):
    #     """Находит точку take profit для long."""
    #     if self._up_take_lines is None:
    #         # Вычисляем коэффициенты уравнения прямой
    #         m = (self.t2up[1] - self.t1up[1]) / (self.t2up[0] - self.t1up[0])
    #         b = self.t1up[1] - m * self.t1up[0]
    #
    #         # Расширяем линию тренда на две длины от t1up до t2up
    #         vline_x = self.t2up[0] + 1 * (self.t2up[0] - self.t1up[0])
    #
    #         # Находим точку пересечения
    #         x_intersect = vline_x
    #         y_intersect_up_take = m * x_intersect + b
    #
    #         self._up_take_lines = (x_intersect, y_intersect_up_take)
    #
    #     return self._up_take_lines
    #
    # @property
    # def take_price(self):
    #     """Считает цену тейка с -10%."""
    #     if self._take_price is None:
    #         take_price_perv = self.up_take_lines[1]
    #         self._take_price = self.t2up[1] + 0.9 * (
    #                 take_price_perv - self.t2up[1])
    #     return self._take_price
    #
    # @property
    # def stop_price(self):
    #     """Считает цену стопа с -50%."""
    #     if self._stop_price is None:
    #         self._stop_price = self.t1up[1] + 0.5 * (
    #                 self.entry_price - self.t1up[1])
    #     return self._stop_price
    #
    # @property
    # def risk_reward_ratio(self):
    #     """Считает соотношение риск/прибыль."""
    #     if self._risk_reward_ratio is None:
    #         potential_risk = self.entry_price - self.stop_price
    #         potential_reward = self.take_price - self.entry_price
    #
    #         if potential_risk != 0:  # Избегаем деления на ноль
    #             self._risk_reward_ratio = potential_reward / potential_risk
    #         else:
    #             self._risk_reward_ratio = float('inf')  # Бесконечность
    #
    #     return self._risk_reward_ratio
    #
    # @property
    # def percent_difference(self):
    #     """Считает разницу между ценами входа и тейка."""
    #     return abs(
    #         ((self.entry_price - self.take_price) / self.entry_price) * 100
    #     )
