import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Line:
    """Класс, в котором происходит
    расчет базовых параметров модели расширения."""
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    @staticmethod
    def calculate(point1, point2):

        """Находим линию между двумя точками"""

        # Находим коэффициенты уравнения прямой, проходящей через две точки
        if point2[0] != point1[0]:
            slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
            intercept = point1[1] - slope * point1[0]

            x = np.linspace(point1[0], point2[0] + (
                    (point2[0] - point1[0]) * 3), num=300)
            y = slope * x + intercept

            return slope, intercept, (x, y)
        else:
            return None, None, None

    @staticmethod
    def correction_LT(df, t3down, t4down, slope, intercept,
                      return_rightmost=False):
        # В начале функции:
        # Если return_rightmost=True, выбираем последний бар в списке пересечений
        # Иначе выбираем первый бар в списке пересечений
        bar_selector = -1 if return_rightmost else 0

        # Ищем т3'
        rightmost_index_t3down1 = t3down[0]
        # Получаем все бары между t3down[0]+1 и t4down[0]-1
        high = df.loc[t3down[0]:t4down[0] - 1, 'high']
        # Вычисляем, какие бары пересекают линию
        intersects = high > (slope * high.index + intercept)
        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = high[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                rightmost_index_t3down1 = intersect_indices[bar_selector]

        return (rightmost_index_t3down1, df.loc[rightmost_index_t3down1, 'high'])

    @staticmethod
    def correction_LC_t4down1(df, t2down, t4down, slope, intercept):

        rightmost_index_t4down1 = t4down[0]

        # Получаем все бары между t2down[0]+1 и t4down[0]
        low = df.loc[t2down[0] + 1:t4down[0], 'low']
        # Вычисляем, какие бары пересекают линию
        intersects = low < (slope * low.index + intercept)
        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = low[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                rightmost_index_t4down1 = intersect_indices[
                    np.argmin(low[intersects])]

        return (rightmost_index_t4down1, df.loc[rightmost_index_t4down1, 'low'])

    @staticmethod
    def correction_LC_t2down1(df, t1down, t2down, t4down):

        leftmost_index_t2down1 = t2down[0]

        # Ищем t2'
        # Получаем все бары между t1down[0]+1 и t2down[0]
        low = df.loc[t1down[0] + 1:t2down[0], 'low']
        # Вычисляем, какие бары пересекают линию
        slope, intercept, _ = Line.calculate((
            t1down[0], df.loc[t1down[0], 'low']),
            t4down)
        intersects = low < (slope * low.index + intercept)
        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = low[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                leftmost_index_t2down1 = intersect_indices[0]
                # Перестраиваем прямую на обновленной leftmost_index_t2down1
                slope, intercept, _ = Line.calculate((
                    leftmost_index_t2down1, df.loc[leftmost_index_t2down1, 'low']),
                    t4down)

                # Получаем все бары между t1down[0]+1 и t2down[0]
                low = df.loc[t1down[0] + 1:t2down[0], 'low']

                # Вычисляем, какие бары пересекают линию
                intersects = low < (slope * low.index + intercept)

                # Если есть пересечения
                if intersects.any():
                    # Выбираем индексы пересечений
                    intersect_indices = low[intersects].index
                    if not intersect_indices.empty:
                        # Возвращает выбранный бар, пересекающий линию
                        leftmost_index_t2down1 = intersect_indices[
                            np.argmin(low[intersects])]

        return (leftmost_index_t2down1, df.loc[leftmost_index_t2down1, 'low'])

    @staticmethod
    def correction_LT_HP(df, t3down, t5down, slope, intercept):

        rightmost_index_t3down1 = t3down[0]
        high = df.loc[t3down[0]:t5down[0] - 1, 'high']
        intersects = high > (slope * high.index + intercept)
        if intersects.any():
            intersect_indices = high[intersects].index
            if not intersect_indices.empty:
                rightmost_index_t3down1 = intersect_indices[-1]
                slope, intercept, _ = Line.calculate((
                    rightmost_index_t3down1,
                    df.loc[rightmost_index_t3down1, 'high']),
                    t5down)

                high = df.loc[t3down[0]:t5down[0] - 1, 'high']
                intersects = high > (slope * high.index + intercept)

                if intersects.any():
                    intersect_indices = high[intersects].index
                    if not intersect_indices.empty:
                        rightmost_index_t3down1 = intersect_indices[-1]

        return (rightmost_index_t3down1, df.loc[rightmost_index_t3down1, 'high'])

    @staticmethod
    def check_line(
            df,
            slope,
            intercept,
            point1,
            point2,
            direction='high'
    ):
        df_between_points = df.loc[
            (df.index > point1[0]) & (df.index < point2[0])
            ]

        if direction == 'high':
            close_between_points = df_between_points.loc[
                df_between_points['high'] > (slope * df_between_points.index
                                             + intercept)
                ]
        else:  # direction == 'low'
            close_between_points = df_between_points.loc[
                df_between_points['low'] < (slope * df_between_points.index
                                            + intercept)
                ]

        return not close_between_points.empty

    @staticmethod
    def cos_sim(slope1, slope2):
        percentage_parallel = round(
            100 * (1 - abs(slope1 - slope2) / max(abs(slope1), abs(slope2))),
            2)
        return percentage_parallel

    @staticmethod
    def cosine_similarity(slope1, slope2):
        vector1 = [1, slope1]
        vector2 = [1, slope2]
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        return dot_product / (norm1 * norm2)

class Point:
    def __init__(self, df, point):
        self.df = df
        self.point = point

    @staticmethod
    def find_LT_break_point(df, t4down, index, slope, intercept):

        LT_break_point = None
        high = df.loc[t4down[0]:index, 'high']
        intersects = high > (slope * high.index + intercept)
        if intersects.any():
            intersect_indices = high[intersects].index
            if not intersect_indices.empty:
                LT_break_point = intersect_indices[0]
                return (LT_break_point, slope * LT_break_point + intercept)
            else:
                return None

    @staticmethod
    def find_LT_break_point_close(df, t4down, index, slope, intercept):

        LT_break_point = None
        close = df.loc[t4down[0]:index, 'close']
        intersects = close > (slope * close.index + intercept)
        if intersects.any():
            intersect_indices = close[intersects].index
            if not intersect_indices.empty:
                LT_break_point = intersect_indices[0]
                return (LT_break_point, slope * LT_break_point + intercept)
            else:
                return None

    @staticmethod
    def find_LC_break_point(df, t4down, index, slope, intercept):

        LC_break_point = None
        low = df.loc[t4down[0]:index, 'low']
        intersects = low < (slope * low.index + intercept)
        if intersects.any():
            intersect_indices = low[intersects].index
            if not intersect_indices.empty:
                LC_break_point = intersect_indices[0]
                return (LC_break_point, slope * LC_break_point + intercept)
            else:
                return None

    @staticmethod
    def find_extreme_point(df: pd.DataFrame, start: tuple[int, float],
                           end: tuple[int, float], column: str,
                           direction: str) -> tuple[int, float]:
        if direction == 'max':
            extreme_index = df.loc[start[0]:end[0], column].idxmin()
        elif direction == 'min':
            extreme_index = df.loc[start[0]:end[0], column].idxmax()
        else:
            raise ValueError(
                f"Invalid direction: {direction}, expected 'max' or 'min'")

        extreme_price = df.loc[extreme_index, column]

        return extreme_index, extreme_price

    @staticmethod
    def is_extreme(df, point1, point2, column):
        data = df.loc[point1[0]:point2[0], column]
        if column == "high":
            return point2[1] == data.min()
        elif column == "low":
            return point2[1] == data.max()
        else:
            raise ValueError("Column should be either 'low' or 'high'")

    @staticmethod
    def find_tangent_point(df, t1down, t2down, t4down, slope, intercept):

        closes = df.loc[t1down[0]: t2down[0], 'low']
        intersects = closes < (slope * closes.index + intercept)
        if intersects.any():
            intersect_indices = closes[intersects].index

            diff_t4down_intercept = t4down[0] - intersect_indices
            angle_denominator = t4down[1] - closes.loc[intersect_indices]

            for intersect_index, angle_denom, diff in zip(intersect_indices,
                                                          angle_denominator,
                                                          diff_t4down_intercept):
                angle = np.arctan2(angle_denom, diff)
                slope_candidate = np.tan(angle)
                intercept_candidate = closes.loc[
                                          intersect_index] - slope_candidate * intersect_index

                closes_right = df.loc[intersect_index + 1: t2down[0], 'low']
                intersects_right = closes_right < (
                        slope_candidate * closes_right.index + intercept_candidate)

                if intersects_right.any():
                    continue

                slope = slope_candidate
                intercept = intercept_candidate
                break

        x = np.linspace(t1down[0], t4down[0] + ((t4down[0] - t1down[0]) * 3), 300)
        y = slope * x + intercept
        LT_down = (x, y)

        return slope, intercept, LT_down

    @staticmethod
    def find_next_point(df: pd.DataFrame, column: str, current_index: int,
                        direction='-1') -> \
            tuple[int, float]:
        """
        Функция для нахождения следующей точки после указанной.
        :param df: Данные в формате pandas DataFrame
        :param column: Столбец, значение которого нужно получить ('close', 'high' и т.д.)
        :param current_index: Индекс текущей точки
        :return: Следующая точка в формате (index, price)
        """
        if direction == '-1':
            next_index = df.index.get_loc(current_index) - 1
            if next_index >= 0:
                next_price = df.iloc[next_index][column]
                return df.index[next_index], next_price
            else:
                raise IndexError("No more data points before the current one.")
        else:
            next_index = df.index.get_loc(current_index) + 1
            if next_index >= 0:
                next_price = df.iloc[next_index][column]
                return df.index[next_index], next_price
            else:
                raise IndexError("No more data points before the current one.")

    def check_line(
            df,
            slope,
            intercept,
            point1,
            point2,
            direction='high'
    ):
        """Проверяем, нет ли свечей, которые пересекают линию."""

        df_between_points = df.loc[
            (df.index >= point1[0]) & (df.index <= point2[0])
            ]

        if direction == 'high':
            close_between_points = df_between_points.loc[
                df_between_points['close'] > (slope * df_between_points.index
                                              + intercept)
                ]
        else:  # direction == 'low'
            close_between_points = df_between_points.loc[
                df_between_points['close'] < (slope * df_between_points.index
                                              + intercept)
                ]

        return not close_between_points.empty

    class DownEXPProperty:
        """Класс, в котором происходит
        расчет базовых параметров модели расширения."""

        def __init__(self, t1down, t2down, t3down, t4down):
            self.t1down = t1down
            self.t2down = t2down
            self.t3down = t3down
            self.t4down = t4down

        @property
        def entry_price(self):
            """Находит точку enter для short.
            Уменьшает цену t2down на 10% относительно расстояния до take_price."""
            if self._entry_price is None:
                self._entry_price = self.t2down[1]

            return self._entry_price
