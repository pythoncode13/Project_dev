from decimal import Decimal, getcontext

import pandas as pd
import matplotlib.pyplot as plt
from pandas import Timestamp
from other_modules.timing_decorator import timing_decorator

from core.models.points.all_points_up import add_t4_to_combinations
from core.models.up_model_property import Line, Point
from core.models.upmodelclass import UpModel


@timing_decorator
def find_start_points_hl_up(df: pd.DataFrame):

    new_combinations = add_t4_to_combinations(df, range_limit=30)

    up_trend_points = []

    for combination in new_combinations:
        t1up = combination[0]
        t2up = combination[1]
        t3up = combination[2]
        t4up = combination[3]

        # # Проверяем, что т3 это мин лоу на участке т3-т4
        # if df.loc[t3up[0]:t4up[0], 'low'].min() < t3up[1]:
        #     continue
        """ --------------- ЛТ ------------------ """
        # проводим линию ЛТ
        slope_LT, intercept_LT, LT_up = Line.calculate(t1up, t3up)
        # валидация
        if Line.check_line(df, slope_LT, intercept_LT, t1up, t4up,
                           direction='low'):
            t3up1 = Line.correction_LT(df, t3up, t4up, slope_LT, intercept_LT,
                          return_rightmost=True)
            slope_LT, intercept_LT, LT_up = Line.calculate(t1up, t3up1)

        """ --------------- ЛЦ ------------------ """
        # проводим линию ЛЦ
        slope_LC, intercept_LC, LC_up = Line.calculate(t2up, t4up)

        # валидация
        if Line.check_line(df, slope_LC, intercept_LC, (t1up[0]+1, 0), t4up,
                           direction='high'):
            t4up1 = Line.correction_LC_t4up1(df, t2up, t4up, slope_LC, intercept_LC)
            t2up1 = Line.correction_LC_t2up1(df, t1up, t2up, t4up1)

            slope_LC, intercept_LC, LC_up = Line.calculate(t2up1, t4up1)

            if Line.check_line(df, slope_LC, intercept_LC, (t1up[0]+1, 0), t4up1,
                               direction='high'):
                continue

        """ --------------- СТ ------------------ """
        # Поиск точки пересечения прямых ЛТ и ЛЦ
        if slope_LT == slope_LC:
            continue

        parallel = Line.cos_sim(slope_LT, slope_LC)
        if parallel >= 30:
            continue
        x_intersect_LC_LT_point = (intercept_LC - intercept_LT) / (slope_LT - slope_LC)
        y_intersect_LC_LT_point = slope_LT * x_intersect_LC_LT_point + intercept_LT
        CP_up_point = (x_intersect_LC_LT_point, y_intersect_LC_LT_point)

        if x_intersect_LC_LT_point >= t4up[0]:
            continue

        """ --------------- АКТИВАЦИЯ МОДЕЛИ ------------------ """
        dist_cp_t4_x2 = t4up[0] + ((t4up[0] - float(x_intersect_LC_LT_point)) * 2)

        upper_limit = min(int(dist_cp_t4_x2), len(df))

        # Находим бар, который пробил уровень т4
        first_bar_above_t4up = None
        t5up = None
        HP_up_point = None
        for i in range(int(t4up[0]), upper_limit):
            if df.loc[i, 'high'] > t4up[1]:
                first_bar_above_t4up = i
                break

        LT_break_point = Point.find_LT_break_point(df, t4up, upper_limit, slope_LT, intercept_LT)

        LT_break_point_close = None
        if LT_break_point:
            LT_break_point_close = Point.find_LT_break_point_close(df, t4up, upper_limit,
                                                       slope_LT, intercept_LT)

        if (
                # Если first_bar_above_t4up существует
                # и LT_break_point не существует
                not LT_break_point and first_bar_above_t4up
                # ИЛИ
                or
                # Если first_bar_above_t4up и LT_break_point оба существуют
                # и индекс first_bar_above_t4up меньше индекса
                # в кортеже LT_break_point
                (
                        LT_break_point and first_bar_above_t4up and first_bar_above_t4up <
                        LT_break_point[0])
        ):

            t5up_index = df.loc[(t4up[0] + 1):first_bar_above_t4up,
                         'low'].idxmin()
            t5up_price = df.loc[t5up_index, 'low']
            if t5up_price >= t4up[1]:
                break
            t5up = (t5up_index, t5up_price)

            # Проверка пересечение тел свечей т2-т5
            t2up_candle = df.loc[t2up[0]]
            t5up_candle = df.loc[t5up[0]]

            t2up_upper_body_edge = max(t2up_candle['open'],
                                       t2up_candle['close'])
            t5up_lower_body_edge = min(t5up_candle['open'],
                                       t5up_candle['close'])

            if t2up_upper_body_edge > t5up_lower_body_edge:
                continue
            else:
                # find HP
                # проводим линию HP
                slope_LT_HP, intercept_LT_HP, LT_up_HP = Line.calculate(t3up, t5up)
                # валидация
                if Line.check_line(df, slope_LT_HP, intercept_LT_HP, t3up, t5up,
                                   direction='low'):
                    t3up1 = Line.correction_LT_HP(df, t3up, t5up, slope_LT_HP,
                                               intercept_LT_HP)
                    slope_LT_HP, intercept_LT_HP, LT_up_HP = Line.calculate(t3up1, t5up)
                # plt.plot(LT_up_HP[0], LT_up_HP[1], ':', color='purple',
                #          linewidth=0.9)

                # Поиск точки пересечения прямых LT_up_HP и ЛЦ
                # if slope_LT_HP == slope_LC:
                #
                # x_intersect_LC_LT_up_HP_point = (intercept_LC - intercept_LT_HP) / (
                #             slope_LT_HP - slope_LC)
                # y_intersect_LC_LT_up_HP_point = slope_LT_HP * x_intersect_LC_LT_up_HP_point + intercept_LT_HP
                #
                # if y_intersect_LC_LT_up_HP_point > t4up[1] + (t4up[1] - t1up[1]) * 5:
                #     continue
                # if x_intersect_LC_LT_up_HP_point < t4up[0]:
                #     HP_up_point = None
                # else:
                #     HP_up_point = (x_intersect_LC_LT_up_HP_point, y_intersect_LC_LT_up_HP_point)
                HP_up_point = None

        LC_break_point = None
        if first_bar_above_t4up:
            LC_break_point = Point.find_LC_break_point(df, t4up, dist_cp_t4_x2, slope_LC,
                                                       intercept_LC)

        point_under_LC = None
        if LC_break_point:
            point_under_LC = Point.find_LT_break_point(df, LC_break_point, dist_cp_t4_x2, slope_LC, intercept_LC)

        """ --------------- УРОВНИ ТЕЙКОВ ------------------ """
        up_take_4_100 = t4up[1] + (t4up[1] - t1up[1])

        # Вычисляем коэффициенты уравнения прямой
        m = (t2up[1] - t1up[1]) / (t2up[0] - t1up[0])
        b = t1up[1] - m * t1up[0]

        # Расширяем линию тренда на две длины от t1up до t2up
        vline_x = t2up[0] + 1 * (t2up[0] - t1up[0])

        # Находим точку пересечения
        x_intersect = vline_x
        y_intersect_up_take = m * x_intersect + b

        up_take_lines = (x_intersect, y_intersect_up_take)

        # Вычисляем коэффициенты уравнения прямой
        m = (t4up[1] - t1up[1]) / (t4up[0] - t1up[0])
        b = t1up[1] - m * t1up[0]

        # Расширяем линию тренда на две длины от t1up до t2up
        vline_x = t4up[0] + 1 * (t4up[0] - t1up[0])

        # Находим точку пересечения
        x_intersect = vline_x
        y_intersect_up_take = m * x_intersect + b

        up_take_100 = t2up[1] + (t2up[1] - t1up[1]) * 1
        up_take_lines1 = (x_intersect, up_take_100)

        dist_cp_t4_x1 = t4up[0] + ((t4up[0] - float(CP_up_point[0])) * 1)
        print(f"up_trend_points Найдено {len(up_trend_points)} комбинаций.")
        up_trend_points.append(UpModel(t1up, t2up, t3up, t4up, t5up, first_bar_above_t4up, up_take_lines1, dist_cp_t4_x2, HP_up_point, LC_break_point, point_under_LC, LT_break_point, CP_up_point, dist_cp_t4_x1, LT_break_point_close, slope_LT, intercept_LT))

    return up_trend_points
