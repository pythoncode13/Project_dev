import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from core.model_utilities.line import Line
from core.model_utilities.point import Point


def calculate_distance(point1: np.array, point2: np.array) -> float:
    """
    Вычисляет евклидово расстояние между двумя точками.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))


def calculate_percentage_difference(val1: float, val2: float) -> float:
    """
    Вычисляет процентное отношение между двумя значениями, учитывая направление изменения.
    """
    return ((val2 - val1) / val1) * 100


def normalize_points(points: list) -> np.array:
    """
    Нормализует координаты точек с использованием MinMaxScaler.
    """
    scaler = MinMaxScaler()
    points_array = np.array(points)
    points_norm = scaler.fit_transform(points_array)

    return points_norm

def compute_tail_to_body_ratio(df, index, tail_type='lower'):
    """Вычисляет соотношение хвоста свечи к телу свечи."""
    row = df.loc[index]

    # Вычисляем тело свечи
    body = abs(row['open'] - row['close'])

    # Вычисляем хвост свечи
    if tail_type == 'lower':
        tail = row['open'] - row['low'] if row['open'] > row[
            'close'] else row['close'] - row['low']
    elif tail_type == 'upper':
        tail = row['high'] - row['open'] if row['open'] < row[
            'close'] else row['high'] - row['close']
    else:
        raise ValueError(
            f"Unknown tail_type: {tail_type}."
            "Choose between 'lower' or 'upper'."
        )

    # Вычисляем соотношение хвоста и тела свечи
    tail_to_body_ratio = tail / body if body != 0 else 0

    return tail_to_body_ratio

def compute_stats(*points):
    """Вычисляет статистические параметры для получаемых точек."""
    # Получаем список y-координат всех точек
    y_values = [float(point[1]) for point in points]

    # Вычисляем среднее значение y-координат
    mean_y = np.mean(y_values)

    # Вычисляем стандартное отклонение y-координат
    std_dev_y = np.std(y_values)

    # Вычисляем отношение стандартного отклонения к среднему значению
    std_dev_y_mean_y = std_dev_y / mean_y

    return std_dev_y, std_dev_y_mean_y

class AdvancedTradeAnalysis:
    def __init__(self, analysis_advanced):
        self.model = analysis_advanced
        self.df = self.model.df
        self.t1up = self.model.t1up
        self.t2up = self.model.t2up
        self.t3up = self.model.t3up
        self.CP_up_point = self.model.CP_up_point
        self.up_enter_point = self.model.up_enter_point
        self.point_t1_t2 = self.model.point_t1_t2
        self.max_distance = self.model.point_t1_t2
        self.intersection_t1_point_t2_t3_lines = self.calculate_intersection_t1_point_t2_t3_lines()

    @property
    def calculate_property(self):
        """
        Вычисляет основные свойства модели.
        """
        start = max(0, self.t1up[0] - 15)
        end = self.t1up[0]

        df_slice_low = self.df.loc[start:end, 'low']
        df_slice_close = self.df.loc[start:end, 'close']

        min_low = df_slice_low.min()
        min_close = df_slice_close.min()
        max_close = df_slice_close.max()

        percentage_difference_min_low_and_t1 = calculate_percentage_difference(
            self.t1up[1], min_low)
        percentage_difference_min_close_and_t1 = calculate_percentage_difference(
            self.t1up[1], min_close)
        percentage_difference_max_close_and_t1 = calculate_percentage_difference(
            self.t1up[1], max_close)
        diff_price_change, length3_point_t1_t2_norm, angle_t2_t3, area_under_curve = self.calculate_additional_properties()

        return (percentage_difference_min_low_and_t1,
                percentage_difference_min_close_and_t1,
                percentage_difference_max_close_and_t1,
                diff_price_change,
                length3_point_t1_t2_norm,
                angle_t2_t3,
                area_under_curve)

    def calculate_additional_properties(self):
        """
        Вычисляет дополнительные свойства модели.
        """
        points_i = [self.t1up, self.t2up, self.t3up, self.up_enter_point]
        t1up_norm, t2up_norm, t3up_norm, up_enter_point_norm = normalize_points(
            points_i)

        price_change_t1_t2 = (self.t2up[1] - self.t2up[1]) / self.t2up[1]
        price_change_t2_t3 = (t3up_norm[1] - self.t2up[1]) / self.t2up[1]

        diff_price_change = abs(price_change_t1_t2 - price_change_t2_t3)

        points_i = [self.CP_up_point, self.t1up, self.t2up, self.t3up,
                    self.up_enter_point, self.point_t1_t2,
                    self.intersection_t1_point_t2_t3_lines]
        (CP_up_point_norm, t1up_norm, t2up_norm, t3up_norm,
         up_enter_point_norm, point_t1_t2_norm,
         intersection_t1_point_t2_t3_lines_norm) = normalize_points(points_i)

        length3_point_t1_t2_norm = calculate_distance(t1up_norm,
                                                      intersection_t1_point_t2_t3_lines_norm)

        angle_t2_t3 = math.atan2(self.t3up[1] - self.t2up[1],
                                 self.t3up[0] - self.t2up[0])
        area_under_curve = 0.5 * (self.t3up[0] - self.t1up[0]) * (
                    self.t1up[1] + self.t3up[1])

        return diff_price_change, length3_point_t1_t2_norm, angle_t2_t3, area_under_curve

    @property
    def calculate_param_t2_t3_up_enter(self):
        """
        Вычисляет угол и радиус кривизны между заданными точками.
        """
        t2up = np.array(self.t2up)
        t3up = np.array(self.t3up)
        up_enter_point = np.array(self.up_enter_point)

        angle_t3_enter = math.atan2(up_enter_point[1] - t3up[1],
                                    up_enter_point[0] - t3up[0])
        radius_curvature_t2_t3_enter = abs(
            (t3up[1] - t2up[1]) * (up_enter_point[0] - t3up[0]) - (
                        t3up[0] - t2up[0]) * (
                        up_enter_point[1] - t3up[1])) / np.sqrt(
            (t3up[0] - t2up[0]) ** 2 + (t3up[1] - t2up[1]) ** 2) ** 3

        return angle_t3_enter, radius_curvature_t2_t3_enter

    @property
    def candle_tail_body_parameters(self):
        self.tail_to_body_ratio_t1 = compute_tail_to_body_ratio(
            self.df, self.t1up[0]
        )
        self.tail_to_body_ratio_t2 = compute_tail_to_body_ratio(
            self.df, self.t2up[0], tail_type='upper'
        )
        self.tail_to_body_ratio_t3 = compute_tail_to_body_ratio(
            self.df, self.t3up[0]
        )
        self.tail_to_body_ratio_enter_point_back_1 = compute_tail_to_body_ratio(
            self.df, self.up_enter_point[0] - 1)

        return self.tail_to_body_ratio_t1, self.tail_to_body_ratio_t2, self.tail_to_body_ratio_t3, self.tail_to_body_ratio_enter_point_back_1

    @property
    def compute_std_dev_y_mean_y(self):
        """Вычисляет статистические параметры для нескольких групп точек."""
        # Вычисляем статистические параметры для различных групп точек
        _, self.std_dev_y_mean_y = compute_stats(self.t1up, self.t2up, self.t3up)
        _, self.std_dev_y_mean_y_1 = compute_stats(self.t1up, self.t2up, self.t3up, self.up_enter_point)
        _, self.std_dev_y_mean_y_2 = compute_stats(self.t1up,
                                              self.intersection_t1_point_t2_t3_lines,
                                              self.point_t1_t2)
        _, self.std_dev_y_mean_y_3 = compute_stats(self.t1up, self.t2up, self.t3up, self.up_enter_point,
                                              self.intersection_t1_point_t2_t3_lines,
                                              self.point_t1_t2)
        _, self.std_dev_y_mean_y_4 = compute_stats(self.t1up, self.t2up, self.t3up)
        _, self.std_dev_y_mean_y_5 = compute_stats(self.t1up, self.t2up, self.up_enter_point)
        _, self.std_dev_y_mean_y_6 = compute_stats(self.t1up, self.t3up,
                                              self.intersection_t1_point_t2_t3_lines)
        _, self.std_dev_y_mean_y_7 = compute_stats(self.t2up, self.up_enter_point, self.point_t1_t2)
        self.std_dev_y_t2_t3_up_enter, _ = compute_stats(self.t2up, self.t3up, self.up_enter_point)

        return (
            self.std_dev_y_mean_y,
            self.std_dev_y_mean_y_1,
            self.std_dev_y_mean_y_2,
            self.std_dev_y_mean_y_3,
            self.std_dev_y_mean_y_4,
            self.std_dev_y_mean_y_5,
            self.std_dev_y_mean_y_6,
            self.std_dev_y_mean_y_7,
            self.std_dev_y_t2_t3_up_enter
        )

    @property
    def get_rsi(self):
        self.rsi_value1 = self.df.loc[self.t1up[0], 'rsi']
        self.rsi_value2 = self.df.loc[self.t2up[0], 'rsi']
        self.rsi_value3 = self.df.loc[self.t3up[0], 'rsi']
        self.rsi_value_enter = self.df.loc[self.up_enter_point[0], 'rsi']

        return (
            self.rsi_value1,
            self.rsi_value2,
            self.rsi_value3,
            self.rsi_value_enter,
        )

    @property
    def get_vwap(self):
        self.vwap_t1 = self.df.loc[self.t1up[0], 'vwap']
        self.vwap_enter = self.df.loc[self.up_enter_point[0], 'vwap']
        self.vwap_ratio_t1 = self.t1up[1] / self.vwap_t1
        self. vwap_ratio_enter = self.up_enter_point[1] / self.vwap_enter

        self.vwap_t1_v2 = self.df.loc[self.t1up[0], 'vwap2']
        self.vwap_enter_v2 = self.df.loc[self.up_enter_point[0], 'vwap2']
        self.vwap_ratio_t1_v2 = self.t1up[1] / self.vwap_t1
        self.vwap_ratio_enter_v2 = self.up_enter_point[1] / self.vwap_enter

        return (
            self.vwap_t1_v2,
            self.vwap_enter_v2,
            self.vwap_ratio_t1_v2,
            self.vwap_ratio_enter_v2,
            self.vwap_ratio_t1,
            self.vwap_ratio_enter,
            )

    def calculate_intersection_t1_point_t2_t3_lines(self):
        # провести прямую т1-т2
        Line_t1_t2 = Line.calculate_1(self.t1up, self.t2up, 0)
        # найти новую точку
        Point_extr_t1_t2, max_distance = Point.find_extreme_bar(self.df, self.t1up[0], self.t2up[0], Line_t1_t2, direction='above')
        # провести прямую Point_extr_t1_t2 - t3
        Line_Point_extr_t1_t2_to_t3 = Line.calculate_1(Point_extr_t1_t2, self.t3up, 0)
        # from decimal import Decimal, getcontext
        #
        # getcontext().prec = 10

        # # Находим коэффициенты уравнения прямой, проходящей через две точки
        #
        # slope = Decimal((self.t3up[1] - Point_extr_t1_t2[1]) / (self.t3up[0] - Point_extr_t1_t2[0]))
        # intercept = Decimal(Point_extr_t1_t2[1] - slope * Point_extr_t1_t2[0])

        # найти пересениче т3-новая точка
        intersection_t1_point_t2_t3_lines = Point.find_intersect_two_line_point(Line_t1_t2.intercept,
                                      Line_t1_t2.slope,
                                      Line_Point_extr_t1_t2_to_t3.intercept,
                                      Line_Point_extr_t1_t2_to_t3.slope)
        # x_intersect = (Line_t1_t2.intercept2 - Line_Point_extr_t1_t2_to_t3.intercept1) / (Line_Point_extr_t1_t2_to_t3.slope1 - Line_t1_t2.slope2)
        # y_intersect = Line_Point_extr_t1_t2_to_t3.slope1 * x_intersect + Line_Point_extr_t1_t2_to_t3.intercept1
        #
        self.intersection_t1_point_t2_t3_lines = intersection_t1_point_t2_t3_lines
        entry_date = self.df.loc[self.up_enter_point[0], 'dateTime']

        return self.intersection_t1_point_t2_t3_lines