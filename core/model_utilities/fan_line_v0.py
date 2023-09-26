import matplotlib.pyplot as plt
import numpy as np
from sympy import Point, Polygon
from decimal import Decimal, getcontext
from itertools import combinations

from core.model_utilities.line import Line
from core.model_utilities.point import Point


class FanLine:
    def __init__(self, model, entry_index):
        # Инициализируем модель
        self.model = model
        self.entry_index = entry_index

        self.x_CP, self.y_CP = self.calculate_CP()

        self.right_limit_line = self.calculate_right_limit_line()
        self.l0_point2 = self.calculate_l0_point2()
        self.lt1_point2 = self.calculate_lt1_point2()
        self.lc1_point2 = self.calculate_lc1_point2()
        self.lc_point2, self.lt_point2 = self.recalculate_lc_and_lt_points()
        self.lt_line = self.recalculate_lt_line()

    def calculate_CP(self):
        # Точка CP(x, y)
        self.x_CP = self.model.CP[0]
        self.y_CP = self.model.CP[1]
        return self.x_CP, self.y_CP

    def calculate_right_limit_line(self):
        return self.model.t3[0] + ((self.model.t3[0] - self.model.t1[0]) * 3)

    def calculate_l0_point2(self):
        # Середина между конечными точками
        x_mid = (self.model.t3[0] + self.model.t4[0]) / 2
        y_mid = (self.model.t3[1] + self.model.t4[1]) / 2

        self.l0_point2 = (x_mid, y_mid)

        return self.l0_point2

    def calculate_l0_line(self):
        l0_line = Line.calculate_fan((self.x_CP, self.y_CP), self.l0_point2,
                                 self.right_limit_line)
        return l0_line

    def calculate_lt1_point2(self):
        # Середины для новых линий
        x_LT1_last = (self.model.t3[0] + self.l0_point2[0]) / 2
        y_LT1_last = (self.model.t3[1] + self.l0_point2[1]) / 2

        self.lt1_point2 = (x_LT1_last, y_LT1_last)

        return self.lt1_point2

    def calculate_lt1_line(self):
        LT1_line = Line.calculate_fan((self.x_CP, self.y_CP),
                                      self.lt1_point2,
                                      self.right_limit_line)
        return LT1_line

    def calculate_lc1_point2(self):
        x_LC1_last = (self.model.t4[0] + self.l0_point2[0]) / 2
        y_LC1_last = (self.model.t4[1] + self.l0_point2[1]) / 2
        self.lc1_point2 = (x_LC1_last, y_LC1_last)

        return self.lc1_point2

    def calculate_lc1_line(self):

        LC1_line = Line.calculate_fan((self.x_CP, self.y_CP),
                                      self.lc1_point2,
                                      self.right_limit_line
                                      )
        return LC1_line

    def recalculate_lc_and_lt_points(self):
        # Используем slope и intercept для вычисления y для точки CP
        y_CP_LC = self.model.LC.slope * self.x_CP + self.model.LC.intercept

        # Получаем индексы верхней линии (LC),
        # которые находятся в диапазоне x для нижней линии (LT)
        indices_LC = [i for i, x in enumerate(self.model.LC.points[0]) if
                      min(self.model.LT.points[0]) <= x <= max(
                          self.model.LT.points[0])]

        # Получаем x и y для верхней линии (LC) в найденном диапазоне,
        # включая точку CP
        x_LC_extended = [self.x_CP,
                         *[self.model.LC.points[0][i] for i in indices_LC]]
        y_LC_extended = [y_CP_LC,
                         *[self.model.LC.points[1][i] for i in indices_LC]]
        x_LC_last, y_LC_last = x_LC_extended[-1], y_LC_extended[-1]
        self.lc_point2 = (x_LC_last, y_LC_last)

        # переопределяем ЛТ
        y_CP_LT = self.model.LT.slope * self.x_CP + self.model.LT.intercept
        # Получаем x и y для нижней линии (LT), включая точку CP
        x_LT_extended = [self.x_CP, *self.model.LT.points[0]]
        y_LT_extended = [y_CP_LT, *self.model.LT.points[1]]
        x_LT_last, y_LT_last = x_LT_extended[-1], y_LT_extended[-1]
        self.lt_point2 = (x_LT_last, y_LT_last)
        lt_line = Line.calculate_1((self.x_CP, self.y_CP),
                                   self.lt_point2, 0)

        return self.lc_point2, self.lt_point2

    def recalculate_lt_line(self):

        self.lt_line = Line.calculate((self.x_CP, self.y_CP), self.lt_point2)

        return self.lt_line

    def recalculate_lc_line(self):

        # Определяем линию
        lc_line = Line.calculate_1((self.x_CP, self.y_CP),
                                   self.lc_point2, 0)

        return lc_line

    @staticmethod
    def fan_plot(lt_line, lc_line, l0_line, LT1_line, LC1_line):
        # Рисуем линии (предположим, что у объекта LineProperties есть свойства x и y)
        # Рисуем нижнюю линию полностью, включая точку CP
        plt.plot(lt_line.points[0], lt_line.points[1], ':', color='purple',
                 linewidth=0.9)

        # Рисуем верхнюю линию только в найденном диапазоне, включая точку CP
        plt.plot(lc_line.points[0], lc_line.points[1], ':', color='purple',
                 linewidth=0.9)

        # plt.plot(l0_line.points[0], l0_line.points[1], '--', color='green',
        #          linewidth=1)
        # plt.plot(LT1_line.points[0], LT1_line.points[1], '--', color='blue',
        #          linewidth=1)
        # plt.plot(LC1_line.points[0], LC1_line.points[1], '--', color='red',
        #          linewidth=1)
        #
        # # Закрасим пространство между LT и LT_1
        # plt.fill_between(LT1_line.points[0], LT1_line.points[1],
        #                  lt_line.points[1][-len(LT1_line.points[1]):],
        #                  color='purple', alpha=0.3)
        #
        # # Закрасим пространство между LC и LC_1
        # plt.fill_between(LC1_line.points[0], LC1_line.points[1],
        #                  lc_line.points[1][-len(LC1_line.points[1]):],
        #                  color='purple', alpha=0.3)

    def build_fan(self):
        lt_line = self.model.LT
        lc_line = self.recalculate_lc_line()
        l0_line = self.calculate_l0_line()
        LT1_line = self.calculate_lt1_line()
        LC1_line = self.calculate_lc1_line()

        # Рисуем веер
        FanLine.fan_plot(lt_line, lc_line, l0_line, LT1_line, LC1_line)

        return l0_line, LT1_line, LC1_line

    def angle_lt_lt1(self):
        angle_lt_lt1 = Line.calculate_angle((self.x_CP, self.y_CP),
                                            self.lt_point2, self.lt1_point2)
        return angle_lt_lt1

    def angle_lc_lc1(self):
        angle_lc_lc1 = Line.calculate_angle((self.x_CP, self.y_CP),
                                            self.lc_point2, self.lc1_point2)
        return angle_lc_lc1

    def angle_lt1_lc1(self):
        angle_lt1_lc1 = Line.calculate_angle((self.x_CP, self.y_CP),
                                             self.lt1_point2, self.lc1_point2)
        return angle_lt1_lc1

    def angle_fan(self):
        angle_lt_lt1 = self.angle_lt_lt1()
        angle_lc_lc1 = self.angle_lc_lc1()
        angle_lt1_lc1 = self.angle_lt1_lc1()

        return (angle_lt_lt1,
                angle_lc_lc1,
                angle_lt1_lc1)

    def build_quadro(self):


        points_i =[
            self.model.CP,
            self.model.t1,
            self.model.t2,
            self.model.t3,
            self.model.t4,

        ]

        (CP_norm, t1_norm, t2_norm, t3_norm,
         t4_norm,) = Point.normalize_points(points_i)

        # y_at_t4 = (self.model.t4[0] - lc_line.intercept) / lt_line.slope
        # self.ax.plot(self.model.t4[0], y_at_t4, marker='o', color='k')

        # Создаем четырехугольник с помощью этих точек
        t3_1 = (t3_norm[0], t1_norm[1])

        t1_t2 = (t1_norm[0], t2_norm[1])
        t3_t2 = (t3_norm[0], t2_norm[1])

        
        # polygon = Polygon(t3_1, t3_t2, t1_t2, t1_norm)


        # Вычисляем площадь четырехугольника
        polygon_quadro_t1_t3 = Polygon(t1_norm, t3_1, t3_t2, t1_t2)
        area_quadro_t1_t3 = round(float(polygon_quadro_t1_t3.area), 3)

        print('area_quadro_t1_t3', area_quadro_t1_t3)
        t4_3 = (t4_norm[0], t3_norm[1])
        t3_4 = (t3_norm[0], t4_norm[1])

        polygon_quadro_t3_t4 = Polygon(t3_1, t4_3, t4_norm, t3_4)
        area_quadro_t3_t4 = round(float(polygon_quadro_t3_t4.area), 3)
        print('area_quadro_t3_t4', area_quadro_t3_t4)

        t1_t4 = (t1_norm[0], t4_norm[1])
        polygon_quadro_t1_t4 = Polygon(t1_norm, t4_3, t4_norm, t1_t4)
        area_quadro_t1_t4 = round(float(polygon_quadro_t1_t4.area), 3)
        print('area_quadro_t1_t4', area_quadro_t1_t4)

        aq_ratio_t1_t4_to_t1_t3 = area_quadro_t1_t4 / area_quadro_t1_t3
        aq_ratio_t3_t4_to_t1_t3 = area_quadro_t3_t4 / area_quadro_t1_t3
        aq_free_area = (area_quadro_t1_t4
                        - (area_quadro_t3_t4 + area_quadro_t1_t3)
                        )

        angle_t3_t4_t4_t3 = Line.calculate_angle(t3_norm, t4_norm, t4_3)

        print('angle_t3_t4_t4_t3', angle_t3_t4_t4_t3)
        ''' ---------------------------------------------------- '''
        # Разница по оси Y (высота)
        height_t1_t4 = t4_norm[1] - t1_norm[1]

        # Разница по оси X (ширина)
        width_t1_t4 = t4_norm[0] - t1_norm[0]

        # Соотношение высоты к ширине
        aspect_ratio_t1_t4 = round((height_t1_t4 / width_t1_t4), 3)

        print('aspect_ratio', aspect_ratio_t1_t4)
        ''' ----------------------------------------------------- '''

        # Разница по оси Y (высота)
        height_t1_t2 = t2_norm[1] - t1_norm[1]

        # Разница по оси X (ширина)
        width_t1_t3 = t3_norm[0] - t1_norm[0]

        # Соотношение высоты к ширине
        aspect_ratio_t1_t3 = round((height_t1_t2 / width_t1_t3), 3)

        print('aspect_ratio_t1_t3', aspect_ratio_t1_t3)

        ''' ----------------------------------------------------- '''

        # Соотношение высоты t1-t2 к высоте t1-t4
        height_ratio_t1_t2_to_t1_t4 = round((height_t1_t2 / height_t1_t4), 3)
        print('height_ratio_t1_t2_to_t1_t4', height_ratio_t1_t2_to_t1_t4)

        width_t3_t4 = t4_norm[0] - t3_norm[0]
        width_ratio_t3_t4_to_t3_t4 = round((width_t3_t4 / width_t1_t3), 3)
        print('width_ratio_t1_t3_to_t3_t4', width_ratio_t3_t4_to_t3_t4)

        width_t1_t2 = t2_norm[0] - t1_norm[0]
        print('width_t1_t2', width_t1_t2)
        print('width_t3_t4', width_t3_t4)

        width_ratio_t1_t2_to_t3_t4 = round((width_t3_t4 / width_t1_t2), 3)
        print('width_ratio_t1_t2_to_t3_t4', width_ratio_t1_t2_to_t3_t4)

        width_t2_t4 = t4_norm[0] - t2_norm[0]
        width_ratio_t1_t2_to_t2_t4 = round((width_t2_t4 / width_t1_t2), 3)
        print('width_ratio_t1_t2_to_t2_t4', width_ratio_t1_t2_to_t2_t4)
        width_t2_t3 = t3_norm[0] - t2_norm[0]

        width_ratio_t1_t2_to_t2_t3 = round((width_t2_t3 / width_t1_t2), 3)
        print('width_ratio_t1_t2_to_t2_t3', width_ratio_t1_t2_to_t2_t3)

        width_t1_t2_t3_t4 = round((width_ratio_t1_t2_to_t3_t4 / width_ratio_t1_t2_to_t2_t3), 3)
        print('width_t1_t2_t3_t4', width_t1_t2_t3_t4)
        width_t2_t3_t4 = round(
            (width_t3_t4 / width_t2_t3), 3)
        print('width_t2_t3_t4', width_t2_t3_t4)

        width_CP_t1 = t1_norm[0] - CP_norm[0]

        width_CP_t1_to_t1_t3 = round(
            (width_t1_t3 / width_CP_t1), 3)

        width_t1_t3_to_t3_t4 = round(
            (width_t3_t4 / width_t1_t3), 3)
        height_t2_t4 = t4_norm[1] - t2_norm[1]
        height_ratio_t1_t2_to_t2_t4 = round((height_t1_t2 / height_t2_t4), 3)
        print('height_ratio_t1_t2_to_t2_t4', height_ratio_t1_t2_to_t2_t4)

        dist_CP_t1_norm = Line.calculate_distance(t1_norm, CP_norm)
        print('dist_CP_t1_norm', dist_CP_t1_norm)

        ratio_dist_CP_t1_t3 = aspect_ratio_t1_t3 / dist_CP_t1_norm

        print('ratio_dist_CP_t1_t3', ratio_dist_CP_t1_t3)
        return (aq_ratio_t1_t4_to_t1_t3,
                aq_ratio_t3_t4_to_t1_t3,
                aq_free_area,
                angle_t3_t4_t4_t3,
                aspect_ratio_t1_t4,
                aspect_ratio_t1_t3,
                height_ratio_t1_t2_to_t1_t4,
                width_ratio_t3_t4_to_t3_t4,
                width_ratio_t1_t2_to_t3_t4,
                width_ratio_t1_t2_to_t2_t4,
                width_ratio_t1_t2_to_t2_t3,
                width_t1_t2_t3_t4,
                width_t2_t3_t4,
                width_CP_t1,
                width_CP_t1_to_t1_t3,
                width_t1_t3_to_t3_t4,
                height_ratio_t1_t2_to_t2_t4,
                dist_CP_t1_norm,
                ratio_dist_CP_t1_t3,
                )

        # print('t1_norm', t1_norm, '\n',
        #       't4_3', t4_3, '\n',
        #       't4_norm', t4_norm, '\n',
        #       't1_t4', t1_t4, '\n', )

        # plt.plot([self.model.t1[0], t1_t2[0]], [self.model.t1[1], t1_t2[1]],
        #          'r-')
        # plt.plot([t1_t2[0], t3_t2[0]], [t1_t2[1], t3_t2[1]], 'r-')
        # plt.plot([t3_t2[0], t3_1[0]], [t3_t2[1], t3_1[1]], 'r-')
        # plt.plot([t3_1[0], self.model.t1[0]], [t3_1[1], self.model.t1[1]],
        #          'r-')

        # plt.plot(self.model.t4[0], self.model.t1[1], marker='o', color='k')

    def dist_lines(self):

        # dist_CP_t1_norm = Line.calculate_distance(t1_norm, CP_norm)

        entry_point = (self.entry_index, self.model.t4[1])

        min_low_index = self.model.df.loc[self.model.t4[0]+1:self.entry_index-1, 'low'].idxmin()
        t5 = (min_low_index, self.model.df.loc[min_low_index, 'low'])
        print('t4[0]', self.model.t4[0])
        print('entry_index', self.entry_index)

        y_at_t1 = float(self.model.LC.slope) * self.model.t1[0] + float(
            self.model.LC.intercept)
        at_1 = (self.model.t1[0], y_at_t1)

        y_at_t2 = float(self.lt_line.slope) * self.model.t2[0] + float(
            self.lt_line.intercept)
        at_2 = (self.model.t2[0], y_at_t2)

        y_at_t2_lc = float(self.model.LC.slope) * self.model.t2[0] + float(
            self.model.LC.intercept)
        at_2_lc = (self.model.t2[0], y_at_t2_lc)

        y_at_t3 = float(self.model.LC.slope) * self.model.t3[0] + float(
            self.model.LC.intercept)
        at_3 = (self.model.t3[0], y_at_t3)
        y_at_t4 = float(self.lt_line.slope) * self.model.t4[0] + float(self.lt_line.intercept)
        at_4 = (self.model.t4[0], y_at_t4)
        y_at_t5 = float(self.lt_line.slope) * t5[0] + float(
            self.lt_line.intercept)
        at_5 = (t5[0], y_at_t5)

        y_at_t5_lc = float(self.model.LC.slope) * t5[0] + float(
            self.model.LC.intercept)
        at_5_lc = (t5[0], y_at_t5_lc)

        y_at_t6 = float(self.lt_line.slope) * self.entry_index + float(
            self.lt_line.intercept)
        at_6 = (self.entry_index, y_at_t6)

        y_at_t6_lc = float(self.model.LC.slope) * self.entry_index + float(
            self.model.LC.intercept)
        at_6_lc = (self.entry_index, y_at_t6_lc)

        mid_dist_t2_t5 = ((self.model.t2[0] + t5[0]) / 2, (self.model.t2[1] + t5[1]) / 2)


        mid_dist_t5_entry = ((t5[0] + entry_point[0]) / 2, (t5[1] + entry_point[1]) / 2)


        mid_dist_t1_entry = (
        (self.model.t1[0] + entry_point[0]) / 2, (self.model.t1[1] + entry_point[1]) / 2)
        print('t1', self.model.t1)
        print('t5', t5)
        print('t1_t2_line')
        (t1_t2_line, above_t1_t2, distance_to_above_t1_t2,  below_t1_t2,
         distance_to_below_t1_t2, sum_dist_t1_t2, ratio_above_below_t1_t2
         ) = FanLine.find_max_distance_bar_above_below_line(self.model.df,
                                                            self.model.t1,
                                                            self.model.t2,
                                                            self.model.t1[0]+1,
                                                            self.model.t2[0]
                                                            )
        print('t2_t3_line')
        (t2_t3_line, above_t2_t3, distance_to_above_t2_t3, below_t2_t3,
         distance_to_below_t2_t3, sum_dist_t2_t3, ratio_above_below_t2_t3
         ) = FanLine.find_max_distance_bar_above_below_line(self.model.df,
                                                            self.model.t2,
                                                            self.model.t3,
                                                            self.model.t2[0],
                                                            self.model.t3[0]
                                                            )
        print('t3_t4_line')
        (t3_t4_line, above_t3_t4, distance_to_above_t3_t4, below_t3_t4,
         distance_to_below_t3_t4, sum_dist_t3_t4, ratio_above_below_t3_t4
         ) = FanLine.find_max_distance_bar_above_below_line(self.model.df,
                                                            self.model.t3,
                                                            self.model.t4,
                                                            self.model.t3[0],
                                                            self.model.t4[0]
                                                            )
        print('t4_t5_line')

        (t4_t5_line, above_t4_t5, distance_to_above_t4_t5, below_t4_t5,
         distance_to_below_t4_t5, sum_dist_t4_t5, ratio_above_below_t4_t5
         ) = FanLine.find_max_distance_bar_above_below_line(self.model.df,
                                                            self.model.t4,
                                                            t5,
                                                            self.model.t4[0],
                                                            t5[0]
                                                            )
        print('t5_entry_line')
        print('t5_entry_line', t5)
        print('entry_point', entry_point)

        (t5_entry_line, above_t5_entry, distance_to_above_t5_entry,
         below_t5_entry, distance_to_below_t5_entry, sum_dist_t5_entry,
         ratio_above_below_t5_entry
         ) = FanLine.find_max_distance_bar_above_below_line(self.model.df,
                                                            t5,
                                                            entry_point,
                                                            t5[0],
                                                            entry_point[0]-1
                                                            )
        print('t2_t5_line')

        (t2_t5_line, above_t2_t5, distance_to_above_t2_t5,
         below_t2_t5, distance_to_below_t2_t5, sum_dist_t2_t5,
         ratio_above_below_t2_t5
         ) = FanLine.find_max_distance_bar_above_below_line(self.model.df,
                                                            self.model.t2,
                                                            t5,
                                                            self.model.t2[0],
                                                            t5[0]
                                                            )
        # t2_t5_line t3_t4_line
        intersect_t3_t4_t2_t5 = Point.find_intersect_two_line_point(t2_t5_line.intercept,
                                                                    t2_t5_line.slope,
                                                                    t3_t4_line.intercept,
                                                                    t3_t4_line.slope)

        print('t3_entry_line')

        (t3_entry_line, above_t3_entry, distance_to_above_t3_entry,
         below_t3_entry, distance_to_below_t3_entry, sum_dist_t3_entry,
         ratio_above_below_t3_entry
         ) = FanLine.find_max_distance_bar_above_below_line(self.model.df,
                                                            self.model.t3,
                                                            entry_point,
                                                            self.model.t3[0],
                                                            entry_point[0]-1
                                                            )
        print('t1_entry_line')

        (t1_entry_line, above_t1_entry, distance_to_above_t1_entry,
         below_t1_entry, distance_to_below_t1_entry, sum_dist_t1_entry,
         ratio_above_below_t1_entry
         ) = FanLine.find_max_distance_bar_above_below_line(self.model.df,
                                                            self.model.t1,
                                                            entry_point,
                                                            self.model.t1[0],
                                                            entry_point[0]-1
                                                            )
        plt.plot(below_t1_entry[0], below_t1_entry[1], 'bo')
        plt.plot(above_t1_entry[0], above_t1_entry[1], 'bo')
        print('t5', t5)
        print('distance_to_above_t1_entry', distance_to_above_t1_entry)
        print('distance_to_below_t1_entry', distance_to_below_t1_entry)

        print('ratio_above_below_t1_entry', ratio_above_below_t1_entry)


        enter_cp = (self.entry_index, self.model.CP[1])
        cp_enter = (self.model.CP[0], entry_point[1])
        t4_cp = (self.model.t4[0], self.model.CP[1])
        t3_cp = (self.model.t3[0], self.model.CP[1])
        t2_cp = (self.model.t2[0], self.model.CP[1])
        t1_cp = (self.model.t1[0], self.model.CP[1])

        take100_point = (self.model.t4[0], self.model.properties.up_take_100)

        points_i = [
            self.model.CP,
            self.model.t1,
            self.model.t2,
            self.model.t3,
            self.model.t4,
            t5,
            entry_point,

            at_1,
            at_2,
            at_2_lc,
            at_3,
            at_4,
            at_5,
            at_5_lc,
            at_6,
            at_6_lc,

            enter_cp,
            cp_enter,
            t4_cp,
            t3_cp,
            t2_cp,
            t1_cp,

            above_t1_t2,
            below_t1_t2,
            above_t2_t3,
            below_t2_t3,
            above_t3_t4,
            below_t3_t4,
            above_t4_t5,
            below_t4_t5,
            above_t5_entry,
            below_t5_entry,
            above_t2_t5,
            below_t2_t5,
            above_t3_entry,
            below_t3_entry,
            above_t1_entry,
            below_t1_entry,

            intersect_t3_t4_t2_t5,
            take100_point,

            mid_dist_t2_t5,
            mid_dist_t5_entry,
            mid_dist_t1_entry,
        ]

        (cp_norm,
         t1_norm,
         t2_norm,
         t3_norm,
         t4_norm,
         t5_norm,
         entry_point_norm,

         at_1_norm,
         at_2_norm,
         at_2_lc_norm,
         at_3_norm,
         at_4_norm,
         at_5_norm,
         at_5_lc_norm,
         at_6_norm,
         at_6_lc_norm,

         enter_cp_norm,
         cp_enter_norm,
         t4_cp_norm,
         t3_cp_norm,
         t2_cp_norm,
         t1_cp_norm,

         above_t1_t2_norm,
         below_t1_t2_norm,
         above_t2_t3_norm,
         below_t2_t3_norm,
         above_t3_t4_norm,
         below_t3_t4_norm,
         above_t4_t5_norm,
         below_t4_t5_norm,
         above_t5_entry_norm,
         below_t5_entry_norm,
         above_t2_t5_norm,
         below_t2_t5_norm,
         above_t3_entry_norm,
         below_t3_entry_norm,
         above_t1_entry_norm,
         below_t1_entry_norm,

         intersect_t3_t4_t2_t5_norm,
         take100_point_norm,
         mid_dist_t2_t5_norm,
         mid_dist_t5_entry_norm,
         mid_dist_t1_entry_norm,
         ) = Point.normalize_points(points_i)

        distances = FanLine.calculate_dist_beetwen_point(
            cp_norm=cp_norm,
            t1_norm=t1_norm,
            t2_norm=t2_norm,
            t3_norm=t3_norm,
            t4_norm=t4_norm,
            t5_norm=t5_norm,
            entry_point_norm=entry_point_norm,
            at_1_norm=at_1_norm,
            at_2_norm=at_2_norm,
            at_2_lc_norm=at_2_lc_norm,
            at_3_norm=at_3_norm,
            at_4_norm=at_4_norm,
            at_5_norm=at_5_norm,
            at_5_lc_norm=at_5_lc_norm,
            at_6_norm=at_6_norm,
            at_6_lc_norm=at_6_lc_norm,
            enter_cp_norm=enter_cp_norm,
            cp_enter_norm=cp_enter_norm,
            t4_cp_norm=t4_cp_norm,
            t3_cp_norm=t3_cp_norm,
            t2_cp_norm=t2_cp_norm,
            t1_cp_norm=t1_cp_norm,
            above_t1_t2_norm=above_t1_t2_norm,
            below_t1_t2_norm=below_t1_t2_norm,
            above_t2_t3_norm=above_t2_t3_norm,
            below_t2_t3_norm=below_t2_t3_norm,
            above_t3_t4_norm=above_t3_t4_norm,
            below_t3_t4_norm=below_t3_t4_norm,
            above_t4_t5_norm=above_t4_t5_norm,
            below_t4_t5_norm=below_t4_t5_norm,
            above_t5_entry_norm=above_t5_entry_norm,
            below_t5_entry_norm=below_t5_entry_norm,
            above_t2_t5_norm=above_t2_t5_norm,
            below_t2_t5_norm=below_t2_t5_norm,
            above_t3_entry_norm=above_t3_entry_norm,
            below_t3_entry_norm=below_t3_entry_norm,
            above_t1_entry_norm=above_t1_entry_norm,
            below_t1_entry_norm=below_t1_entry_norm,
            intersect_t3_t4_t2_t5_norm=intersect_t3_t4_t2_t5_norm,
            take100_point_norm=take100_point_norm,
            mid_dist_t2_t5_norm=mid_dist_t2_t5_norm,
            mid_dist_t5_entry_norm=mid_dist_t5_entry_norm,
            mid_dist_t1_entry_norm=mid_dist_t1_entry_norm,
        )
        print('distances')
        print(distances)

        # print(below_t4_t5_norm, t1_norm)

        # cp_dist1 = Line.calculate_distance(cp_norm, t1_norm)
        # cp_dist2 = Line.calculate_distance(cp_norm, t2_norm)
        # cp_dist3 = Line.calculate_distance(cp_norm, t3_norm)
        # cp_dist4 = Line.calculate_distance(cp_norm, t4_norm)
        # cp_dist5 = Line.calculate_distance(cp_norm, t5_norm)
        # cp_dist6 = Line.calculate_distance(cp_norm, entry_point_norm)
        #
        # dist_t2_t4 = Line.calculate_distance(t2_norm, t4_norm)
        # print('dist_t2_t4', dist_t2_t4)
        # dist_t4_take100_point = Line.calculate_distance(t4_norm, take100_point_norm)
        # print('dist_t4_take100_point', dist_t4_take100_point)


        # # сперва нормализовать
        # # расстояние между двумя точками над и под линией т1-т2
        # dist_above_below_t1_t2 = Line.calculate_distance(above_t1_t2,
        #                                                  below_t1_t2)

        # plt.plot(t5[0], t5[1], marker='o', color='g')
        # plt.annotate('t5', t5)
        # plt.plot(entry_point[0], entry_point[1], marker='o', color='k')
        # plt.annotate('entry_point', entry_point)
        # plt.plot(at_1[0], at_1[1], marker='o', color='g')
        # plt.annotate('at_1', at_1)
        # plt.plot(at_2[0], at_2[1], marker='o', color='g')
        # plt.annotate('at_2', at_2)
        # plt.plot(at_2_lc[0], at_2_lc[1], marker='o', color='g')
        # plt.annotate('at_2_lc', at_2_lc)
        # plt.plot(at_3[0], at_3[1], marker='o', color='b')
        # plt.annotate('at_3', at_3)
        # plt.plot(at_4[0], at_4[1], marker='o', color='r')
        # plt.annotate('at_4', at_4)
        # plt.plot(at_5[0], at_5[1], marker='o', color='y')
        # plt.annotate('at_5', at_5)
        # plt.plot(at_5_lc[0], at_5_lc[1], marker='o', color='y')
        # plt.annotate('at_5_lc', at_5_lc)
        # plt.plot(at_6[0], at_6[1], marker='o', color='k')
        # plt.annotate('at_6', at_6)
        # plt.plot(at_6_lc[0], at_6_lc[1], marker='o', color='k')
        # plt.annotate('at_6_lc', at_6_lc)
        #
        # plt.plot(intersect_t3_t4_t2_t5[0], intersect_t3_t4_t2_t5[1], marker='o', color='k')
        # plt.annotate('intersect_t3_t4_t2_t5', intersect_t3_t4_t2_t5)
        #
        # plt.plot(enter_cp[0], enter_cp[1],
        #          marker='o', color='k')
        # plt.annotate('enter_cp', enter_cp)
        # plt.plot(cp_enter[0], cp_enter[1],
        #          marker='o', color='k')
        # plt.annotate('cp_enter', cp_enter)
        # plt.plot(t4_cp[0], t4_cp[1],
        #          marker='o', color='k')
        # plt.annotate('t4_cp', t4_cp)
        # plt.plot(t3_cp[0], t3_cp[1],
        #          marker='o', color='k')
        # plt.annotate('t3_cp', t3_cp)
        # plt.plot(t2_cp[0], t2_cp[1],
        #          marker='o', color='k')
        # plt.annotate('t2_cp', t2_cp)
        # plt.plot(t1_cp[0], t1_cp[1],
        #          marker='o', color='k')
        # plt.annotate('t1_cp', t1_cp)
        #
        # plt.plot(mid_dist_t2_t5[0], mid_dist_t2_t5[1],
        #          marker='o', color='k')
        # plt.annotate('mid_dist_t2_t5', mid_dist_t2_t5)
        # plt.plot(mid_dist_t5_entry[0], mid_dist_t5_entry[1],
        #          marker='o', color='k')
        # plt.annotate('mid_dist_t5_entry', mid_dist_t5_entry)
        # plt.plot(mid_dist_t1_entry[0], mid_dist_t1_entry[1],
        #          marker='o', color='k')
        # plt.annotate('mid_dist_t1_entry', mid_dist_t1_entry)

        return distances

    @staticmethod
    def calculate_dist_beetwen_point(**kwargs):
        points_dict = kwargs

        # Словарь для хранения расстояний
        distances = {}

        # Используем `combinations` для получения всех уникальных пар точек
        for point1_name, point2_name in combinations(points_dict.keys(), 2):
            # Получаем координаты точек
            point1_coord = points_dict[point1_name]
            point2_coord = points_dict[point2_name]

            # Формируем имя переменной для расстояния
            dist_var_name = f"dist_{point1_name}_{point2_name}"

            # Вычисляем расстояние и сохраняем его в словаре
            distances[dist_var_name] = Line.calculate_distance(point1_coord,
                                                               point2_coord)
        return distances

    @staticmethod
    def find_max_distance_bar_above_below_line(df, point1, point2, index_start, index_end):
        # Формируем линию, соединяющую т1 с т2 - "t1_t2_line"
        point1_point2_line = Line.calculate_1(point1, point2)

        # plt.plot(point1_point2_line.points[0],
        #          point1_point2_line.points[1], ':',
        #          color='purple',
        #          linewidth=0.9)

        # Находим точку, которая на макс расстоянии выше от "t1_t2_line" -
        # "t1_t2_above"
        above_p1_p2, distance_to_above_p1_p2 = Point.find_extreme_bar(
            df, index_start, index_end,
            point1_point2_line, direction='above')

        below_p1_p2, distance_to_below_p1_p2 = Point.find_extreme_bar(
            df, index_start, index_end,
            point1_point2_line,
            direction='below')
        # Сумма расстояний от прямой. Можно рассматривать как волатильность,
        # разбег вокруг прямой.
        sum_dist_p1_p2 = (
                    distance_to_above_p1_p2 + distance_to_below_p1_p2)

        # Соотношение расстояниий верхнего к нижнему
        if distance_to_below_p1_p2 != 0:
            ratio_above_below_p1_p2 = np.round((distance_to_above_p1_p2 / distance_to_below_p1_p2), 3)
        else:
            ratio_above_below_p1_p2 = distance_to_above_p1_p2

        return (point1_point2_line,
                above_p1_p2,
                distance_to_above_p1_p2,
                below_p1_p2,
                distance_to_below_p1_p2,
                sum_dist_p1_p2,
                ratio_above_below_p1_p2
                )
