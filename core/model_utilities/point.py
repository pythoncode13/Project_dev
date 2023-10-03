import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Point:
    def __init__(self, df, point):
        self.df = df
        self.point = point

    @staticmethod
    def find_intersect_two_line_point(intercept_LC,
                                      slope_LC,
                                      intercept_LT,
                                      slope_LT):
        """Находит точку пересечения двух прямых."""

        x_intersect_LC_LT_point = (intercept_LC - intercept_LT) / (
                    slope_LT - slope_LC)
        y_intersect_LC_LT_point = (slope_LT
                                   * x_intersect_LC_LT_point
                                   + intercept_LT
                                   )

        return x_intersect_LC_LT_point, y_intersect_LC_LT_point

    @staticmethod
    def find_first_bar_by_price(df, target_price, point1, point2,
                                direction='above'):
        # Проверяем алиасы для направления
        if direction in ['above', 'up_model']:
            price_column = 'high'
            comparison_operator = np.greater
        elif direction in ['below', 'down_model']:
            price_column = 'low'
            comparison_operator = np.less
        else:
            raise ValueError(
                "Invalid direction. Use 'above', 'below', "
                "'up_model', or 'down_model'.")

        # Создаем массив из выбранного столбца, начиная с индекса point1
        prices = df.loc[point1:point2, price_column].values
        # Ищем индексы в зависимости от выбранного оператора сравнения
        target_price_indices = \
        np.where(comparison_operator(prices, target_price))[0]
        # Если нет таких индексов, возвращаем None
        if target_price_indices.size == 0:
            return None
        # Берем первый индекс, если есть хотя бы один
        first_bar_by_target_price_index = target_price_indices[0] + int(point1)

        first_bar_by_price = (first_bar_by_target_price_index, target_price)

        return first_bar_by_price

    # @staticmethod
    # def find_LT_break_point_close(df, t4up, index, slope, intercept):
    #     """Находит точку достижения ЛТ "клоусом" бара."""
    #     LT_break_point = None
    #     # Получаем все бары между t3up[0]+1 и t4up[0]-1
    #     close = df.loc[t4up[0]:index, 'close']
    #     # Вычисляем, какие бары пересекают линию
    #     intersects = close < (slope * close.index + intercept)
    #     # Если есть пересечения
    #     if intersects.any():
    #         # Выбираем индексы пересечений
    #         intersect_indices = close[intersects].index
    #         if not intersect_indices.empty:
    #             # Возвращает выбранный бар, пересекающий линию
    #             LT_break_point = intersect_indices[0]
    #             return (LT_break_point, slope * LT_break_point + intercept)
    #         else:
    #             return None
    @staticmethod
    def find_LT_break_point_close(df, t4, index, slope, intercept, direction='up_model'):
        """Находит точку достижения ЛТ "клоусом" бара."""

        LT_break_point = None

        # Выбираем столбец и направление сравнения в зависимости от направления
        price_column = 'close'
        comparison_operator = np.less if direction == 'up_model' else np.greater

        # Получаем все бары между t4[0] и index
        prices = df.loc[t4[0]:index, price_column]

        # Вычисляем, какие бары пересекают линию
        intersects = comparison_operator(prices,
                                         slope * prices.index + intercept)

        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = prices[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                LT_break_point = intersect_indices[0]
                return (LT_break_point, slope * LT_break_point + intercept)
        else:
            return None

    @staticmethod
    def find_LT_break_point(df, t4, index, slope, intercept, direction='up_model'):
        """Находит точку достижения ЛТ "лоем" или "хаем" бара."""
        LT_break_point = None

        # Выбираем столбец и направление сравнения в зависимости от направления
        price_column = 'low' if direction == 'up_model' else 'high'
        comparison_operator = np.less if direction == 'up_model' else np.greater

        # Получаем все бары между t4[0] и index
        prices = df.loc[t4[0]:index, price_column]

        # Вычисляем, какие бары пересекают линию
        intersects = comparison_operator(prices,
                                         slope * prices.index + intercept)

        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = prices[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                LT_break_point = intersect_indices[0]
                return (LT_break_point, slope * LT_break_point + intercept)
        else:
            return None

    @staticmethod
    def find_line_break_point_close(df, index1, index2, slope, intercept,
                                  direction='above'):
        """Находит точку выше или ниже прямой "клоусом" бара."""

        # Выбираем столбец и направление сравнения в зависимости от направления
        price_column = 'close'
        comparison_operator = np.greater if direction == 'above' else np.less

        # Получаем все бары между t4[0] и index
        prices = df.loc[index1:index2, price_column]

        # Вычисляем, какие бары пересекают линию
        intersects = comparison_operator(prices,
                                         slope * prices.index + intercept)

        # Если есть пересечения
        if intersects.any():
            # Выбираем индексы пересечений
            intersect_indices = prices[intersects].index
            if not intersect_indices.empty:
                # Возвращает выбранный бар, пересекающий линию
                line_break_point = intersect_indices[0]
                return (line_break_point, slope * line_break_point + intercept)
        else:
            return None

    @staticmethod
    def find_t5(df, t2, t4, first_bar_by_price, direction='up_model'):
        """Находит и проверяет т5 для up_model или down_model."""

        # Выбираем столбец и сравнение в зависимости от направления
        price_column = 'low' if direction == 'up_model' else 'high'
        comparison_operator = np.greater_equal if direction == 'up_model' else np.less_equal

        # Находим индекс минимального 'low' или максимального 'high' между t4[0] + 1 и first_bar_by_price
        t5_index = df.loc[(t4[0] + 1):first_bar_by_price,
                   price_column].idxmin() if direction == 'up_model' else df.loc[(t4[0] + 1):first_bar_by_price, price_column].idxmax()
        t5_price = df.loc[t5_index, price_column]

        # Проверяем условие в зависимости от направления
        if comparison_operator(t5_price, t4[1]):
            return None

        t5 = (t5_index, t5_price)

        # Проверяем t2 и t5 (эта часть не меняется)
        if not Point.check_t2_t5(df, t2, t5, direction=direction):
            return None
        else:
            return t5

    @staticmethod
    def check_t2_t5(df, t2, t5, direction='up_model'):
        """Проверяет наличие пересечения телами свечей т2 и т5
        для up_model или down_model."""
        if t5 is None:
            return False
        else:
            # Проверка пересечение тел свечей т2-т5 в зависимости от направления
            t2_candle = df.loc[t2[0]]
            t5_candle = df.loc[t5[0]]

            if direction == 'up_model':
                t2_edge = max(t2_candle['open'], t2_candle['close'])
                t5_edge = min(t5_candle['open'], t5_candle['close'])
            else:  # down_model
                t2_edge = min(t2_candle['open'], t2_candle['close'])
                t5_edge = max(t5_candle['open'], t5_candle['close'])

            if (direction == 'up_model' and t2_edge < t5_edge) or \
                    (direction == 'down_model' and t2_edge > t5_edge):
                return True

        return False

    @staticmethod
    def normalize_points(points: list) -> np.array:
        """
        Нормализует координаты точек с использованием MinMaxScaler.
        """
        scaler = MinMaxScaler()
        points_array = np.array(points)
        points_norm = scaler.fit_transform(points_array)
        # Округление до 5 знаков после запятой
        points_norm = np.round(points_norm, 5)
        return points_norm

    @staticmethod
    def find_extreme_bar(df, point1, point2, line, direction='above'):
        """
        Находит индекс и расстояние бара, наиболее удаленного от прямой.

        Parameters:
        df (DataFrame): DataFrame с данными.
        point1 (int): Начальная точка диапазона для поиска.
        point2 (int): Конечная точка диапазона для поиска.
        line (object): Объект, содержащий атрибуты slope и intercept прямой.
        direction (str): Направление ('above' или 'below') для поиска бара.

        Returns:
        int: Индекс бара.
        float: Расстояние от бара до прямой.
        """

        # Ограничиваем DataFrame диапазоном от point1 до point2
        filtered_df = df.loc[int(point1):int(point2)].copy()

        slope = float(line.slope)
        intercept = float(line.intercept)

        # 1. Создаем столбец с значениями прямой
        filtered_df['line_values'] = filtered_df.index * slope + intercept

        # Вычисляем расстояния от баров до прямой
        if direction == 'above':
            filtered_df = filtered_df[
                filtered_df['high'] > filtered_df['line_values']]
            filtered_df['distance_to_line'] = np.abs(
                filtered_df['high'] - filtered_df['line_values'])
        elif direction == 'below':
            filtered_df = filtered_df[
                filtered_df['low'] < filtered_df['line_values']]
            filtered_df['distance_to_line'] = np.abs(
                filtered_df['low'] - filtered_df['line_values'])

        # Находим индекс бара с максимальным расстоянием и соответствующую цену
        if not filtered_df['distance_to_line'].empty:
            max_distance_index = filtered_df['distance_to_line'].idxmax()
            price_type = 'high' if direction == 'above' else 'low'
            max_distance_price = np.round(
                filtered_df.loc[max_distance_index, price_type], 7)

            # Формируем точку
            point = (max_distance_index, max_distance_price)

            max_distance = np.round(
                filtered_df.loc[max_distance_index, 'distance_to_line'], 7)
        else:
            # point_index = int((point1 + point2) / 2)
            # point_price_high = np.round(
            #     df.loc[point_index, 'high'], 5)
            # point_price_low = np.round(
            #     df.loc[point_index, 'low'], 5)
            # point_price = np.round((point_price_high / point_price_low), 5)
            # point = (point_index, point_price)
            # max_distance = 0
            point = (point1[0], df.loc[point1[0], 'high'])
            max_distance = np.round(
                filtered_df.loc[point1[0], 'distance_to_line'], 7)

        return point, max_distance
