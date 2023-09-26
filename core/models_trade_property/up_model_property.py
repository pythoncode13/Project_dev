import logging

import numpy as np


def compute_line(point1, point2, resolution=300):
    """Вычисляет уравнение прямой и угол наклона между двумя точками."""
    # Находим угол наклона линии между двумя точками
    angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])

    # Преобразуем угол из радиан в градусы
    angle_deg = np.degrees(angle)

    # Находим коэффициенты уравнения прямой, проходящей через две точки
    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    intercept = point1[1] - slope * point1[0]

    # Получаем координаты прямой
    x = np.linspace(point1[0], point2[0], resolution)
    y = slope * x + intercept

    return angle_deg, slope, intercept, (x, y)


def find_intersection(slope1, intercept1, slope2, intercept2):
    """
    Находит точку пересечения двух прямых,
    заданных их углами наклона и точками пересечения с осью ординат.
    """
    x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
    y_intersect = slope1 * x_intersect + intercept1

    return x_intersect, y_intersect


class UpModelProperty:
    def __init__(self, df, up_enter_point, analysis_base):
        self.df = df
        self.up_enter_point = up_enter_point
        self.analysis_base = analysis_base
        self.t1up = self.analysis_base.t1up
        self.t2up = self.analysis_base.t2up
        self.t3up = self.analysis_base.t3up
        # self.up_take_lines = self.analysis_base.up_take_lines

        self.angle_deg_LT, self.slope_LT, self.intercept_LT, self.LT_up = self.get_LT_up
        self.angle_deg_LC, self.slope_LC, self.intercept_LC, self.LC_up = self.get_LC_up
        self.angle_deg_Line_t1_t2, self.slope_Line_t1_t2, self.intercept_Line_t1_t2, self.Line_t1_t2 = self.get_Line_t1_t2
        self.max_distance, self.point_t1_t2 = self.find_point_t1_t2
        self.angle_deg_Line_t1_point_t2_t3, self.slope_Line_t1_point_t2_t3, self.intercept_Line_t1_point_t2_t3, self.Line_t1_point_t2_t3 = self.get_Line_t1_point_t2_t3
        self.intersection_t1_point_t2_t3_lines = self.find_intersection_t1_point_t2_t3_lines
        self.CP_up_point = self.find_CP_up_point

    """LineModel"""
    @property
    def get_LT_up(self):
        """Получает ЛТ_ап и сопутствующие параметры."""
        self.angle_deg_LT, self.slope_LT, self.intercept_LT, self.LT_up = compute_line(
            self.t1up, self.t3up)
        return self.angle_deg_LT, self.slope_LT, self.intercept_LT, self.LT_up

    @property
    def get_LC_up(self):
        """Получает ЛЦ_ап и сопутствующие параметры."""
        self.angle_deg_LC, self.slope_LC, self.intercept_LC, self.LC_up = compute_line(
            self.t2up,
            self.up_enter_point)
        return self.angle_deg_LC, self.slope_LC, self.intercept_LC, self.LC_up

    @property
    def get_Line_t1_t2(self):
        """Получает линию между точками т1-т2 и сопутствующие параметры."""
        self.angle_deg_Line_t1_t2, self.slope_Line_t1_t2, self.intercept_Line_t1_t2, self.Line_t1_t2 = compute_line(
            self.t1up, self.t2up)
        return self.angle_deg_Line_t1_t2, self.slope_Line_t1_t2, self.intercept_Line_t1_t2, self.Line_t1_t2

    @property
    def find_point_t1_t2(self):
        """Находит максимальное расстояние от точки до линии т1-т2."""

        # Инициализируем самое большое расстояние нулем
        self.max_distance = 0

        self.point_t1_t2 = (self.t1up[0], self.df.loc[self.t1up[0], 'high'])

        # Вычисляем y для x = t1up_x на линии Line_t1_t2
        y_line_at_t1up_x = self.slope_Line_t1_t2 * self.point_t1_t2[
            0] + self.intercept_Line_t1_t2

        # Вычисляем расстояние между t1up и линией Line_t1_t2
        distance_t1up_line = abs(self.point_t1_t2[1] - y_line_at_t1up_x)
        self.max_distance = distance_t1up_line

        # max_distance_bar_index = бар, находящийся на самом
        # большом расстоянии от линии Line_t1_t2

        return self.max_distance, self.point_t1_t2

    @property
    def get_Line_t1_point_t2_t3(self):
        """
        Получает линию между найденной точкой - лоу бара на максимальном
        расстоянии от линии т1-т2 слева и сопутствующие параметры.
        """
        self.angle_deg_Line_t1_point_t2_t3, self.slope_Line_t1_point_t2_t3, self.intercept_Line_t1_point_t2_t3, self.Line_t1_point_t2_t3 = compute_line(
            self.t3up, self.point_t1_t2)
        return self.angle_deg_Line_t1_point_t2_t3, self.slope_Line_t1_point_t2_t3, self.intercept_Line_t1_point_t2_t3, self.Line_t1_point_t2_t3


    """PointModel"""
    @property
    def find_CP_up_point(self):
        """Поиск СТ."""
        self.CP_up_point = find_intersection(self.slope_LT, self.intercept_LT,
                                             self.slope_LC, self.intercept_LC)
        return self.CP_up_point

    @property
    def find_intersection_t1_point_t2_t3_lines(self):
        """
        Поиск пересечения двух прямых: прямая между т1-т2 и
        прямой между дополнительной точкой и т3.
        """
        self.intersection_t1_point_t2_t3_lines = find_intersection(
            self.slope_Line_t1_t2, self.intercept_Line_t1_t2,
            self.slope_Line_t1_point_t2_t3,
            self.intercept_Line_t1_point_t2_t3)
        return self.intersection_t1_point_t2_t3_lines
