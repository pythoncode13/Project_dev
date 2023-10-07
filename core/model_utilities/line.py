import numpy as np
from decimal import Decimal, getcontext

getcontext().prec = 10


class LineProperties:
    def __init__(self, slope, intercept, points):
        self.slope = slope
        self.intercept = intercept
        self.points = points


class Line:
    """Класс, в котором происходит
    расчет базовых параметров модели расширения."""
    def __init__(self):
        pass

    @staticmethod
    def calculate(point1, point2):
        """Находим линию между двумя точками"""

        point1 = (float(point1[0]), float(point1[1]))
        point2 = (float(point2[0]), float(point2[1]))

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

            return LineProperties(slope, intercept, (x, y))
        else:
            return None

    @staticmethod
    def calculate_1(point1, point2, right_edge_index):
        """Находим линию между двумя точками"""

        # Находим коэффициенты уравнения прямой, проходящей через две точки
        if point2[0] != point1[0]:

            point1 = (float(point1[0]), float(point1[1]))
            point2 = (float(point2[0]), float(point2[1]))

            slope = Decimal((point2[1] - point1[1]) / (point2[0] - point1[0]))

            # Применяем Decimal только к этой строке
            # print('point1[0]', point1[0])
            intercept = Decimal(point1[1]) - slope * Decimal(point1[0])

            x = np.linspace(point1[0],
                            point2[0]+right_edge_index, num=300)
            x = [Decimal(xi) for xi in x]  # Преобразовываем x в Decimal
            y = [slope * xi + intercept for xi in
                 x]  # Вычисляем y с использованием Decimal

            return LineProperties(slope, intercept, (x, y))
        else:
            return None

    @staticmethod
    def calculate_fan(point1, point2, right_limit_line):
        """Находим линию между двумя точками"""

        # Находим коэффициенты уравнения прямой, проходящей через две точки
        if point2[0] != point1[0]:

            point1 = (float(point1[0]), float(point1[1]))
            point2 = (float(point2[0]), float(point2[1]))
            right_limit_line = (float(right_limit_line))

            slope = Decimal((point2[1] - point1[1]) / (point2[0] - point1[0]))

            # Применяем Decimal только к этой строке
            # print('point1[0]', point1[0])
            intercept = Decimal(point1[1]) - slope * Decimal(
                point1[0])

            x = np.linspace(point1[0],
                            right_limit_line, num=300)
            x = [Decimal(xi) for xi in x]  # Преобразовываем x в Decimal
            y = [slope * xi + intercept for xi in
                 x]  # Вычисляем y с использованием Decimal

            return LineProperties(slope, intercept, (x, y))
        else:
            return None

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

    # @staticmethod
    # def correction_LC_t4up1(df,
    #                         t2up,
    #                         t4up,
    #                         slope,
    #                         intercept):
    #     """Коррекция на ЛЦ на участке т2-т4.
    #     Добавил Decimal."""
    #     getcontext().prec = 10
    #     slope = Decimal(slope)
    #     intercept = Decimal(intercept)
    #
    #     leftmost_index_t4up1 = t4up[0]
    #
    #     # Получаем все бары между t3up[0]+1 и t4up[0]-1
    #     high = df.loc[t2up[0] + 1:t4up[0], 'high']
    #
    #     # Вычисляем, какие бары пересекают линию
    #     intersects = high > (slope * high.index.map(Decimal) + intercept)
    #
    #     # Если есть пересечения
    #     if intersects.any():
    #         # Выбираем индексы пересечений
    #         intersect_indices = high[intersects].index
    #         if not intersect_indices.empty:
    #             # Возвращает выбранный бар, пересекающий линию
    #             leftmost_index_t4up1 = intersect_indices[np.argmax(high[intersects])]
    #
    #     return leftmost_index_t4up1, df.loc[leftmost_index_t4up1, 'high']

    @staticmethod
    def correction_LC_t4up1(df,
                            t2,
                            t4
                            ):
        # Получаем все бары между t1[0]+1 и t2[0]
        high = df.loc[t2[0] + 1:t4[0], 'high']

        # Сортируем индексы по возрастанию "low"
        sorted_high_indices = high.sort_values().index

        for high_index in sorted_high_indices:
            # Построим линию через эту точку и t4
            LC = Line.calculate(t2, (high_index, high[high_index]))

            # Вычисляем, какие бары пересекают линию
            intersects = high > (LC.slope * high.index + LC.intercept)

            # Если нет пересечений, возвращаем эту точку
            if not intersects.any():
                return (high_index, high[high_index])

        # Если не найдено подходящей точки, возвращаем исходную t2
        return t4


    @staticmethod
    def correction_LC_t2up1(df, t1up, t2up, t4up):
        """Коррекция на ЛЦ на участке т1-т2.
        Добавил Decimal."""

        rightmost_index_t2up1 = t2up[0]

        # Ищем т2'
        # Получаем все бары между t3up[0]+1 и t4up[0]-1
        high = df.loc[t1up[0] + 1:t2up[0], 'high']
        # Вычисляем, какие бары пересекают линию
        LC = Line.calculate((
            t1up[0], df.loc[t1up[0], 'low']),
            t4up)
        intersects = high > (LC.slope * high.index + LC.intercept)

        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = high[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                rightmost_index_t2up1 = intersect_indices[0]

                # Перестраиваем прямую на обновленной rightmost_index_t2up1
                LC_correct = Line.calculate((
                rightmost_index_t2up1, df.loc[rightmost_index_t2up1, 'high']),
                t4up)

                # Получаем все бары между t3up[0]+1 и t4up[0]-1
                high = df.loc[t1up[0] + 1:t2up[0], 'high']

                # Вычисляем, какие бары пересекают линию
                intersects = high > (
                        LC_correct.slope * high.index + LC_correct.intercept
                )

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

    @staticmethod
    def calculate_t3_t2_taget_line(t3, t2):
        """Находит точку take profit для long."""

        # Вычисляем коэффициенты уравнения прямой
        m = (t3[1] - t2[1]) / (t3[0] - t2[0])
        b = t2[1] - m * t2[0]

        # Расширяем линию тренда на две длины от t1up до t2up
        vline_x = t2[0] - 2 * (t3[0] - t2[0])

        # Находим точку пересечения
        x_intersect = vline_x
        y_intersect_up_take = m * x_intersect + b

        up_take_lines = (x_intersect, y_intersect_up_take)

        return up_take_lines


    """ For down model """
    @staticmethod
    def correction_LT_down(df, t3, t4, slope, intercept,
                      return_rightmost=False):
        # В начале функции:
        # Если return_rightmost=True, выбираем последний бар в списке пересечений
        # Иначе выбираем первый бар в списке пересечений
        bar_selector = -1 if return_rightmost else 0

        # Ищем т3'
        rightmost_index_t3_1 = t3[0]
        # Получаем все бары между t3[0]+1 и t4[0]-1
        high = df.loc[t3[0]:t4[0] - 1, 'high']
        # Вычисляем, какие бары пересекают линию
        intersects = high > (slope * high.index + intercept)
        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = high[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                rightmost_index_t3_1 = intersect_indices[bar_selector]

        return (rightmost_index_t3_1, df.loc[rightmost_index_t3_1, 'high'])

    # @staticmethod
    # def correction_LC_t4_1_down(df, t2, t4, slope, intercept):
    #     """Коррекция на ЛЦ на участке т2-т4.
    #     Добавил Decimal."""
    #     getcontext().prec = 10
    #     slope = Decimal(slope)
    #     intercept = Decimal(intercept)
    #
    #     leftmost_index_t4_1 = t4[0]
    #
    #     # Получаем все бары между t3up[0]+1 и t4[0]-1
    #     low = df.loc[t2[0] + 1:t4[0], 'high']
    #
    #     # Вычисляем, какие бары пересекают линию
    #     intersects = low < (slope * low.index.map(Decimal) + intercept)
    #
    #     # Если есть пересечения
    #     if intersects.any():
    #         # Выбираем индексы пересечений
    #         intersect_indices = low[intersects].index
    #         if not intersect_indices.empty:
    #             # Возвращает выбранный бар, пересекающий линию
    #             leftmost_index_t4_1 = intersect_indices[
    #                 np.argmax(low[intersects])]
    #
    #     return leftmost_index_t4_1, df.loc[leftmost_index_t4_1, 'low']

    @staticmethod
    def correction_LC_t4_1_down(df, t2, t4):
        # Получаем все бары между t1[0]+1 и t2[0]
        low = df.loc[t2[0] + 1:t4[0], 'low']

        # Сортируем индексы по возрастанию "low"
        sorted_low_indices = low.sort_values().index

        for low_index in sorted_low_indices:
            # Построим линию через эту точку и t4
            LC = Line.calculate(t2, (low_index, low[low_index]))

            # Вычисляем, какие бары пересекают линию
            intersects = low < (LC.slope * low.index + LC.intercept)

            # Если нет пересечений, возвращаем эту точку
            if not intersects.any():
                return (low_index, low[low_index])

        # Если не найдено подходящей точки, возвращаем исходную t2
        return t4

    @staticmethod
    def correction_LC_t2_1_down(df, t1, t2, t4):
        # Получаем все бары между t1[0]+1 и t2[0]
        low = df.loc[t1[0] + 1:t2[0], 'low']

        # Сортируем индексы по возрастанию "low"
        sorted_low_indices = low.sort_values().index

        for low_index in sorted_low_indices:
            # Построим линию через эту точку и t4
            LC = Line.calculate((low_index, low[low_index]), t4)

            # Вычисляем, какие бары пересекают линию
            intersects = low < (LC.slope * low.index + LC.intercept)

            # Если нет пересечений, возвращаем эту точку
            if not intersects.any():
                return (low_index, low[low_index])

        # Если не найдено подходящей точки, возвращаем исходную t2
        return t2

    @staticmethod
    def cos_sim_down(slope1, slope2):
        """Вычисляем косинусное сходство,
                для определения "параллельности линий."""
        percentage_parallel = round(
            100 * (1 - abs(slope1 - slope2) / max(abs(slope1), abs(slope2))),
            2)
        return percentage_parallel

    @staticmethod
    def calculate_angle_two_points(point1, point2):
        """Вычисляет уравнение прямой и угол наклона между двумя точками."""
        # Находим угол наклона линии между двумя точками
        angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])

        # Преобразуем угол из радиан в градусы
        angle_deg = np.degrees(angle)

        return angle_deg

    @staticmethod
    def calculate_angle(point1, point2, point3):
        """Вычисляет угол в градусах между тремя точками."""
        point1 = (float(point1[0]), float(point1[1]))
        point2 = (float(point2[0]), float(point2[1]))
        point3 = (float(point3[0]), float(point3[1]))

        # Вычисляем векторы A и B
        vector_A = np.array(point2) - np.array(point1)
        vector_B = np.array(point3) - np.array(point1)

        # Вычисляем скалярное произведение и длины векторов
        dot_product = np.dot(vector_A, vector_B)
        magnitude_A = np.linalg.norm(vector_A)
        magnitude_B = np.linalg.norm(vector_B)

        # Вычисляем косинус угла
        cos_theta = dot_product / (magnitude_A * magnitude_B)

        # Вычисляем угол в радианах
        theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        # Переводим угол в градусы
        theta_deg = np.degrees(theta_rad)

        theta_deg = np.round(theta_deg, 1)

        return theta_deg
    @staticmethod
    def lt_lc_for_plot(CP, t3, t4):
        """Пересчет координат линий
        для более удобного отображения на графике."""

        # Задаем правую границу прямой
        right_edge_index_lc = (t4[0] * 2)
        right_edge_index_lt = (right_edge_index_lc
                               + (t4[0] - t3[0])
                               )
        # Пересчитываем значения прямых
        lt = Line.calculate_1(CP, t3, right_edge_index_lt)
        lc = Line.calculate_1(CP, t4, right_edge_index_lc)

        return lt, lc