import numpy as np
import pandas as pd
from decimal import Decimal, getcontext


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
            slope = Decimal((point2[1] - point1[1]) / (point2[0] - point1[0]))

            # Применяем Decimal только к этой строке
            # print('point1[0]', point1[0])
            intercept = Decimal(float(point1[1])) - slope * Decimal(float(point1[0]))

            x = np.linspace(point1[0],
                            point2[0] + ((point2[0] - point1[0]) * 3), num=300)
            x = [Decimal(xi) for xi in x]  # Преобразовываем x в Decimal
            y = [slope * xi + intercept for xi in
                 x]  # Вычисляем y с использованием Decimal

            return slope, intercept, (x, y)
        else:
            return None, None, None

    # @staticmethod
    # def calculate_with_correction(df, point1, point2):
    #     """Находим линию между двуми точками
    #     с учетом пересечения соседними барами."""
    #     # Находим коэффициенты уравнения прямой, проходящей через две точки
    #     if point2[0] != point1[0]:
    #         slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    #         intercept = point1[1] - slope * point1[0]
    #
    #         # Проверка на пересечение следующего или предыдущего бара
    #         dist_t1_t2 = point2[0] - point1[0]
    #         # dist_t2_t4 = point4[0] - point2[0]
    #         next_point_y = df.loc[point1[0] + 1, 'high'] if point1[
    #                                                             0] + 1 in df.index else None
    #         prev_point_y = df.loc[point2[0] - 1, 'high'] if point2[
    #                                                             0] - 1 in df.index else None
    #
    #         if next_point_y is not None and next_point_y > slope * (
    #                 point1[0] + 1) + intercept:
    #             point1 = (point1[0] + 1, next_point_y)
    #             slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    #             intercept = point1[1] - slope * point1[0]
    #         elif prev_point_y is not None and prev_point_y > slope * (
    #                 point2[0] - 1) + intercept:
    #             point2 = (point2[0] - 1, prev_point_y)
    #             slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    #             intercept = point1[1] - slope * point1[0]
    #
    #         x = np.linspace(point1[0],
    #                         point2[0] + ((point2[0] - point1[0]) * 3), num=300)
    #         y = slope * x + intercept
    #         return slope, intercept, (x, y)
    #     else:
    #         return None, None, None

    # @staticmethod
    # def correction_LC(df, t1up, t2up, t3up, t4up, slope, intercept):
    #
    #     # Ищем т2'
    #     leftmost_index_t2up1 = t2up[0]
    #     # Получаем все бары между t3up[0]+1 и t4up[0]-1
    #     high = df.loc[t1up[0] + 1:t2up[0] - 1, 'high']
    #     # Вычисляем, какие бары пересекают линию
    #     intersects = high > (slope * high.index + intercept)
    #     # Если есть пересечения
    #     if intersects.any():
    #         # Выбираем индексы пересечений
    #         intersect_indices = high[intersects].index
    #
    #         if not intersect_indices.empty:
    #             # Возвращает самый левый бар, пересекающий линию
    #             leftmost_index_t2up1 = intersect_indices[0]
    #
    #
    #     # Ищем т4'
    #     leftmost_index_t4up1 = t4up[0]
    #     # Получаем все бары между t3up[0]+1 и t4up[0]-1
    #     high = df.loc[t3up[0] + 1:t4up[0] - 1, 'high']
    #     # Вычисляем, какие бары пересекают линию
    #     intersects = high > (slope * high.index + intercept)
    #     # Если есть пересечения
    #     if intersects.any():
    #         # Выбираем индексы пересечений
    #         intersect_indices = high[intersects].index
    #
    #         if not intersect_indices.empty:
    #             # Возвращает самый левый бар, пересекающий линию
    #             leftmost_index_t4up1 = intersect_indices[0]
    #
    #     return (
    #         leftmost_index_t2up1, df.loc[leftmost_index_t2up1, 'high']), (
    #         leftmost_index_t4up1, df.loc[leftmost_index_t4up1, 'high'])

        # slope = (t4up1[1] - t2up1[1]) / (t4up1[0] - t2up1[0])
        # intercept = t2up1[1] - slope * t2up1[0]
        #
        # x = np.linspace(t2up1[0], t4up1[0] + (
        #         (t4up1[0] - t2up1[0]) * 3), num=300)
        # y = slope * x + intercept
        # return slope, intercept, (x, y)


        # else:
        #     # Если нет пересечений, нет баров, пересекающих линию, возвращаем None или другое значение по вашему усмотрению
        #     return slope, intercept, LC_up

    @staticmethod
    def correction_LT(df, t3up, t4up, slope, intercept,
                      return_rightmost=False):
        # В начале функции:
        # Если return_rightmost=True, выбираем последний бар в списке пересечений
        # Иначе выбираем первый бар в списке пересечений
        bar_selector = -1 if return_rightmost else 0

        # Ищем т3'
        leftmost_index_t3up1 = t3up[0]
        # Получаем все бары между t3up[0]+1 и t4up[0]-1
        low = df.loc[t3up[0]:t4up[0]-1, 'low']
        # Вычисляем, какие бары пересекают линию
        intersects = low < (slope * low.index + intercept)
        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = low[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                leftmost_index_t3up1 = intersect_indices[bar_selector]

        return (leftmost_index_t3up1, df.loc[leftmost_index_t3up1, 'low'])



    @staticmethod
    def correction_LC_t4up1(df,
                            t2up,
                            t4up,
                            slope,
                            intercept):
        """Коррекция на ЛЦ на участке т2-т4.
        Добавил Decimal."""
        getcontext().prec = 10
        slope = Decimal(slope)
        intercept = Decimal(intercept)

        leftmost_index_t4up1 = t4up[0]

        # Получаем все бары между t3up[0]+1 и t4up[0]-1
        high = df.loc[t2up[0] + 1:t4up[0], 'high']

        # Вычисляем, какие бары пересекают линию
        intersects = high > (slope * high.index.map(Decimal) + intercept)

        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = high[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                leftmost_index_t4up1 = intersect_indices[np.argmax(high[intersects])]

        return leftmost_index_t4up1, df.loc[leftmost_index_t4up1, 'high']


    @staticmethod
    def correction_LC_t2up1(df, t1up, t2up, t4up):
        """Коррекция на ЛЦ на участке т1-т2.
        Добавил Decimal."""

        rightmost_index_t2up1 = t2up[0]

        # Ищем т2'
        # Получаем все бары между t3up[0]+1 и t4up[0]-1
        high = df.loc[t1up[0] + 1:t2up[0], 'high']
        # Вычисляем, какие бары пересекают линию
        slope, intercept, _ = Line.calculate((
            t1up[0], df.loc[t1up[0], 'low']),
            t4up)
        intersects = high > (slope * high.index + intercept)

        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = high[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                rightmost_index_t2up1 = intersect_indices[0]

                # Перестраиваем прямую на обновленной rightmost_index_t2up1
                slope, intercept, _ = Line.calculate((
                rightmost_index_t2up1, df.loc[rightmost_index_t2up1, 'high']),
                t4up)

                # Получаем все бары между t3up[0]+1 и t4up[0]-1
                high = df.loc[t1up[0] + 1:t2up[0], 'high']

                # Вычисляем, какие бары пересекают линию
                intersects = high > (slope * high.index + intercept)

                # Если есть пересечения
                if intersects.any():
                    # Выбираем индексы пересечений
                    intersect_indices = high[intersects].index
                    if not intersect_indices.empty:
                        # Возвращает выбранный бар, пересекающий линию
                        rightmost_index_t2up1 = intersect_indices[
                            np.argmax(high[intersects])]

        return (rightmost_index_t2up1, df.loc[rightmost_index_t2up1, 'high'])

    @staticmethod
    def correction_LT_HP(df, t3up, t5up, slope, intercept):
        # В начале функции:
        # Если return_rightmost=True, выбираем последний бар в списке пересечений
        # Иначе выбираем первый бар в списке пересечений

        # Ищем т3'
        leftmost_index_t3up1 = t3up[0]
        # Получаем все бары между t3up[0]+1 и t4up[0]-1
        low = df.loc[t3up[0]:t5up[0] - 1, 'low']
        # Вычисляем, какие бары пересекают линию
        intersects = low < (slope * low.index + intercept)
        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = low[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                leftmost_index_t3up1 = intersect_indices[-1]
                # Перестраиваем прямую на обновленной rightmost_index_t2up1
                slope, intercept, _ = Line.calculate((
                    leftmost_index_t3up1,
                    df.loc[leftmost_index_t3up1, 'low']),
                    t5up)

                # Получаем все бары между t3up[0]+1 и t4up[0]-1
                low = df.loc[t3up[0]:t5up[0] - 1, 'low']

                # Вычисляем, какие бары пересекают линию
                intersects = low < (slope * low.index + intercept)

                # Если есть пересечения
                if intersects.any():
                    # Выбираем индексы пересечений
                    intersect_indices = low[intersects].index
                    if not intersect_indices.empty:
                        # Возвращает выбранный бар, пересекающий линию
                        leftmost_index_t3up1 = intersect_indices[-1]

        return (leftmost_index_t3up1, df.loc[leftmost_index_t3up1, 'low'])

    @staticmethod
    def check_line(df, slope, intercept, point1, point2, direction='low'):
        """Проверяем, нет ли свечей, которые пересекают линию."""

        df_between_points = df.loc[
            (df.index > point1[0]) & (df.index < point2[0])
            ]

        if direction == 'low':
            close_between_points = df_between_points.loc[
                df_between_points['low'] < (slope * df_between_points.index
                                              + intercept)
                ]
        else:  # direction == 'high'
            close_between_points = df_between_points.loc[
                df_between_points['high'] > (slope * df_between_points.index
                                              + intercept)
                ]

        return not close_between_points.empty

    @staticmethod
    def cos_sim(slope1, slope2):
        """Вычисляет процент параллельности."""
        percentage_parallel = round(100 * (
                    1 - abs(slope1 - slope2) / max(slope1, slope2)), 2)

        return percentage_parallel

    @staticmethod
    def cosine_similarity(slope1, slope2):
        """Вычисляем косинусное сходство,
        для определения "параллельности линий."""

        # Преобразовываем углы наклона в векторы
        vector1 = [1, slope1]
        vector2 = [1, slope2]

        # Вычисляем скалярное произведение
        dot_product = np.dot(vector1, vector2)

        # Вычисляем нормы векторов
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        # Вычисляем и возвращаем косинусное сходство
        return dot_product / (norm1 * norm2)


class Point:
    def __init__(self, df, point):
        self.df = df
        self.point = point
    #
    # @property
    # def index(self):
    #     return self.point[0]
    #
    # @property
    # def price(self):
    #     return self.df.loc[self.index, 'close']

    @staticmethod
    def find_LT_break_point(df, t4up, index, slope, intercept):

        LT_break_point = None
        # Получаем все бары между t3up[0]+1 и t4up[0]-1
        low = df.loc[t4up[0]:index, 'low']
        # Вычисляем, какие бары пересекают линию
        intersects = low < (slope * low.index + intercept)
        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = low[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                LT_break_point = intersect_indices[0]
                return (LT_break_point, slope * LT_break_point + intercept)
            else:
                return None

    @staticmethod
    def find_LT_break_point_close(df, t4up, index, slope, intercept):

        LT_break_point = None
        # Получаем все бары между t3up[0]+1 и t4up[0]-1
        close = df.loc[t4up[0]:index, 'close']
        # Вычисляем, какие бары пересекают линию
        intersects = close < (slope * close.index + intercept)
        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = close[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                LT_break_point = intersect_indices[0]
                return (LT_break_point, slope * LT_break_point + intercept)
            else:
                return None

    @staticmethod
    def find_LC_break_point(df, t4up, index, slope, intercept):

        LC_break_point = None
        # Получаем все бары между t3up[0]+1 и t4up[0]-1
        high = df.loc[t4up[0]:index, 'high']
        # Вычисляем, какие бары пересекают линию
        intersects = high > (slope * high.index + intercept)
        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = high[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                LC_break_point = intersect_indices[0]
                return (LC_break_point, slope * LC_break_point + intercept)
            else:
                return None

    @staticmethod
    def find_extreme_point(df: pd.DataFrame, start: tuple[int, float],
                           end: tuple[int, float], column: str,
                           direction: str) -> tuple[int, float]:
        """
        Функция для нахождения бара с максимальным или минимальным значением в заданном столбце между двумя точками.
        :param df: Данные в формате pandas DataFrame
        :param start: Начальная точка в формате (index, price)
        :param end: Конечная точка в формате (index, price)
        :param column: Столбец, в котором ищется экстремум ('close', 'high' и т.д.)
        :param direction: Направление поиска ('max' для максимума, 'min' для минимума)
        :return: Точка экстремума в формате (index, price)
        """
        if direction == 'max':
            extreme_index = df.loc[start[0]:end[0], column].idxmax()
        elif direction == 'min':
            extreme_index = df.loc[start[0]:end[0], column].idxmin()
        else:
            raise ValueError(
                f"Invalid direction: {direction}, expected 'max' or 'min'")

        extreme_price = df.loc[extreme_index, column]

        return extreme_index, extreme_price

    @staticmethod
    def is_extreme(df, point1, point2, column):
        # Получаем данные между point1 и point2
        data = df.loc[point1[0]:point2[0], column]

        # Если column равен "low", то проверяем, является ли point2 минимальным на участке
        if column == "low":
            return point2[1] == data.min()
        # Если column равен "high", то проверяем, является ли point2 максимальным на участке
        elif column == "high":
            return point2[1] == data.max()
        else:
            raise ValueError("Column should be either 'low' or 'high'")


    @staticmethod
    def find_tangent_point(df, t1up, t2up, t4up, slope, intercept):

        # Находим индексы баров между t1 и t2
        closes = df.loc[t1up[0]: t2up[0], 'high']

        # Вычисляем пересечения
        intersects = closes > (slope * closes.index + intercept)

        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = closes[intersects].index

            diff_t4up_intercept = t4up[0] - intersect_indices
            angle_denominator = t4up[1] - closes.loc[intersect_indices]

            for intersect_index, angle_denom, diff in zip(intersect_indices,
                                                          angle_denominator,
                                                          diff_t4up_intercept):
                # Вычисляем новые углы и коэффициенты
                angle = np.arctan2(angle_denom, diff)
                slope_candidate = np.tan(angle)
                intercept_candidate = closes.loc[
                                          intersect_index] - slope_candidate * intersect_index

                # Проверяем, есть ли свечи справа, которые пересекают прямую
                closes_right = df.loc[intersect_index + 1: t2up[0], 'high']
                intersects_right = closes_right > (
                            slope_candidate * closes_right.index + intercept_candidate)

                # Если есть бары справа, которые пересекают линию, пропускаем текущий индекс пересечения
                if intersects_right.any():
                    continue

                # Если нет баров справа, которые пересекают линию, принимаем текущие slope и intercept
                slope = slope_candidate
                intercept = intercept_candidate
                break

        # Получаем переменную прямой
        x = np.linspace(t1up[0], t4up[0] + ((t4up[0] - t1up[0]) * 3), 300)
        y = slope * x + intercept
        LT_up = (x, y)

        # Возвращаем точку, которая продлевает линию, и прямую
        # (t1up[0] + 1, slope * (t1up[0] + 1) + intercept)
        return slope, intercept, LT_up

    @staticmethod
    def find_next_point(df: pd.DataFrame, column: str, current_index: int, direction='+1') -> \
    tuple[int, float]:
        """
        Функция для нахождения следующей точки после указанной.
        :param df: Данные в формате pandas DataFrame
        :param column: Столбец, значение которого нужно получить ('close', 'high' и т.д.)
        :param current_index: Индекс текущей точки
        :return: Следующая точка в формате (index, price)
        """
        if direction == '+1':
            next_index = df.index.get_loc(current_index) + 1
            if next_index < len(df):
                next_price = df.iloc[next_index][column]
                return df.index[next_index], next_price
            else:
                raise IndexError("No more data points after the current one.")
        else:
            next_index = df.index.get_loc(current_index) - 1
            if next_index < len(df):
                next_price = df.iloc[next_index][column]
                return df.index[next_index], next_price
            else:
                raise IndexError("No more data points after the current one.")

def check_line(
        df,
        slope,
        intercept,
        point1,
        point2,
        direction='low'
):
    """Проверяем, нет ли свечей, которые пересекают линию."""

    df_between_points = df.loc[
        (df.index >= point1[0]) & (df.index <= point2[0])
    ]

    if direction == 'low':
        close_between_points = df_between_points.loc[
            df_between_points['close'] < (slope * df_between_points.index
                                          + intercept)
        ]
    else:  # direction == 'high'
        close_between_points = df_between_points.loc[
            df_between_points['close'] > (slope * df_between_points.index
                                          + intercept)
        ]

    return not close_between_points.empty


class UpEXPProperty:
    """Класс, в котором происходит
    расчет базовых параметров модели расширения."""
    def __init__(self, t1up, t2up, t3up, t4up):
        self.t1up = t1up
        self.t2up = t2up
        self.t3up = t3up
        self.t4up = t4up



    @property
    def entry_price(self):
        """Находит точку enter для long.
        Увеличивает цену t2up на 10% относительно расстояния до take_price."""
        if self._entry_price is None:

            self._entry_price = self.t2up[1]

        return self._entry_price

    @property
    def find_dist_cp_t4_x1(self):
        if self.dist_cp_t4_x1 is None:
            self.dist_cp_t4_x1 = self.t4up[0] + (self.t4up[0] - self.CP_up_point[0])
        return self.dist_cp_t4_x1

    @property
    def find_dist_cp_t4_x2(self):
        if self.dist_cp_t4_x2 is None:
            self.dist_cp_t4_x2 = self.t4up[0] + ((
                    self.t4up[0] - self.CP_up_point[0]) * 2)
        return self.dist_cp_t4_x2

