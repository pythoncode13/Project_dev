import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from core.up_models.UpCloseModels import UpATR, UpMDB, UpEXP
# from core.up_models.up_exp_property import UpEXPProperty
from core.models.up_model_property import Line, Point
from core.models.upmodelclass import UpModel

def find_start_points_hl(
        combinations: list[tuple[int, float]],
        df: pd.DataFrame
):

    up_trend_points = []

    # Перебираем комбинации точек, удовлетворяющие условию
    for combination in combinations:
        t1up, t2up, t3up, t4up = combination
        # plt.plot(t1up[0], t1up[1], 'bo', color='k')

        # if df.loc[t1up[0]]['low'] != min(df.loc[t1up[0] - 3: t1up[0]]['low']):
        #     continue
        if t1up[0] > t2up[0] > t3up[0]:
            continue
        if min(t1up[1], t2up[1], t3up[1]) != t1up[1]:
            continue
        if t1up[1] > t3up[1] or t3up[1] > t2up[1]:
            continue
        if df.loc[t1up[0]]['low'] != min(df.loc[t1up[0]:t3up[0]]['low']):
            continue
        if df.loc[t2up[0]]['high'] != max(df.loc[t1up[0]:t3up[0]]['high']):
            continue
        if df.loc[t3up[0]]['low'] != min(df.loc[t2up[0]:t3up[0]]['low']):
            continue
        if t1up[0] == t2up[0] or t2up[0] == t3up[0]:
            continue

        # проводим линию ЛТ
        slope_LT, intercept_LT, LT_up = Line.calculate(t1up, t3up)
        # валидация
        if Line.check_line(df, slope_LT, intercept_LT, t1up, t3up,
                           direction='low'):
            continue
        if t4up[0] < t3up[0]:
            continue
        if t4up[1] < t3up[1]:
            continue
        if df.loc[t4up[0]]['high'] != max(df.loc[t1up[0]:t4up[0]]['high']):
            continue
        # валидация
        if Line.check_line(df, slope_LT, intercept_LT, t1up, t4up,
                           direction='low'):
            continue

        # # проводим линию ЛЦ
        slope_LC, intercept_LC, LC_up = Line.calculate(t2up, t4up)
        # валидация
        if Line.check_line(df, slope_LC, intercept_LC, t2up, t4up,
                           direction='high'):
            continue

        # if Line.check_line(df, slope_LC, intercept_LC, t1up, t4up,
        #                     direction='high'):
        #     slope_LC_new, intercept_LC_new, LС_up_new = Point.find_tangent_point(df, t1up, t2up, t4up, slope_LC, intercept_LC)
        #     slope_LC = slope_LC_new
        #     intercept_LC = intercept_LC_new
        #     LC_up = LС_up_new
        #
        # x_intersect_LC_LT_point = (intercept_LC - intercept_LT) / (slope_LT - slope_LC)
        # y_intersect_LC_LT_point = slope_LT * x_intersect_LC_LT_point + intercept_LT
        #
        # if x_intersect_LC_LT_point <= t2up[0]:
        #     CP_up_point = (x_intersect_LC_LT_point, y_intersect_LC_LT_point)
        # else:
        #     continue
        #
        # # Проверяем, где точка пересечения находится относительно t1up и t4up
        # if slope_LT == slope_LC == 0 or Line.cosine_similarity(slope_LT, slope_LC) > 0.9995:
        #     continue

        # """"""
        # # Константа числа Фи
        # phi = 1.618
        # tolerance = 10
        # # Вычисляем длины отрезков (в данном случае мы предполагаем, что t1up, t2up и t3up являются координатами)
        # price_length_long = abs(
        #     t2up[1] - t1up[1])  # длина длинного отрезка t1up:t2up
        # price_length_short = abs(
        #     t2up[1] - t3up[1])  # длина короткого отрезка t2up:t3up
        #
        # # Вычисляем длины отрезков (в данном случае мы предполагаем, что t1up, t2up и t3up являются координатами)
        # index_length_long = abs(
        #     t2up[0] - t1up[0])  # длина длинного отрезка t1up:t2up
        # index_length_short = abs(
        #     t3up[0] - t2up[0])  # длина короткого отрезка t2up:t3up
        #
        # # Проверяем соотношение длин
        # index_ratio = index_length_long / index_length_short
        # price_ratio = price_length_long / price_length_short
        # # Проверяем близость к числу Фи
        # if abs(index_ratio - phi) >= tolerance:
        #     continue
        # if abs(price_ratio - phi) >= tolerance:
        #     continue

        plt.plot(t1up[0], t1up[1], 'o', color='k')
        plt.text(t1up[0], t1up[1], t1up[0], fontsize=10)
        plt.plot(t2up[0], t2up[1], 'yo')
        plt.plot(t3up[0], t3up[1], 'bo')

        plt.plot(LT_up[0], LT_up[1], ':', color='purple',
                 linewidth=0.9)
        plt.plot(LC_up[0], LC_up[1], ':', color='purple',
                 linewidth=0.9)

        up_trend_points.append(UpModel(
            df,
            t1up,
            t2up,
            t3up,
            t4up,
            LC_up,
            slope_LC,
            intercept_LC,
            LT_up,
            slope_LT,
            intercept_LT,
            # CP_up_point
        ))

        print(f'Количество найденных комбинаций close/close: {len(up_trend_points)}')

    return up_trend_points

    # if df.loc[t2up[0]]['close'] != max(df.loc[t1up[0]:t3up[0]]['close']):
    #     continue

    # if t1up[0] > t2up[0] > t3up[0]:
    #     continue
    # if min(t1up[1], t2up[1], t3up[1]) != t1up[1]:
    #     continue
    # if t1up[1] > t3up[1] or t3up[1] > t2up[1]:
    #     continue
    # if df.loc[t1up[0]]['close'] != min(df.loc[t1up[0]:t3up[0]]['close']):
    #     continue
    # if df.loc[t2up[0]]['close'] != max(df.loc[t1up[0]:t3up[0]]['close']):
    #     continue
    # if df.loc[t3up[0]]['close'] != min(df.loc[t2up[0]:t3up[0]]['close']):
    #     continue
    # if t1up[0] == t2up[0] or t2up[0] == t3up[0]:
    #     continue
    #
    # # проводим линию ЛТ
    # slope_LT, intercept_LT, LT_up = Line.calculate(t1up, t3up)
    # # валидация
    # if Line.check_line(df, slope_LT, intercept_LT, t1up, t3up, direction='low'):
    #     continue
    # if t4up[0] < t3up[0]:
    #     continue
    # if t4up[1] < t3up[1]:
    #     continue
    # if df.loc[t4up[0]]['close'] != max(df.loc[t1up[0]:t4up[0]]['close']):
    #     continue
    # # валидация
    # if Line.check_line(df, slope_LT, intercept_LT, t1up, t4up, direction='low'):
    #     continue
    #
    # # проводим линию ЛЦ
    # slope_LC, intercept_LC, LC_up = Line.calculate(t2up, t4up)
    # # валидация
    # if Line.check_line(df, slope_LC, intercept_LC, t2up, t4up, direction='high'):
    #     continue
    #
    # # Находим точку пересечения двух линий
    # x_intersect_LC_LT_point = (intercept_LC - intercept_LT) / (slope_LT - slope_LC)
    # y_intersect_LC_LT_point = slope_LT * x_intersect_LC_LT_point + intercept_LT
    #
    # # Определяем параллельность линий
    # cos_similarity = cosine_similarity(slope_LT, slope_LC)
    # print('t1up[0]:', t1up[0])
    # print('cos_similarity:', cos_similarity)
    # # Проверяем, где точка пересечения находится относительно t1up и t4up
    # if slope_LT == slope_LC == 0 or cos_similarity > 0.996:
    #     up_MDB.append(UpMDB(df, t1up, t2up, t3up, t4up, LT_up, LC_up))
    # elif x_intersect_LC_LT_point <= t2up[0]:
    #     CP_up_point = (
    #     x_intersect_LC_LT_point, y_intersect_LC_LT_point)
    #     up_EXP.append(UpEXP(df, t1up, t2up, t3up, t4up, LT_up, LC_up,
    #                         CP_up_point))
    # elif x_intersect_LC_LT_point >= t4up[0]:
    #     HP_up_point = (
    #     x_intersect_LC_LT_point, y_intersect_LC_LT_point)

    #
    #     # Находим точку пересечения двух линий
    #     x_intersect_LC_LT_point = (intercept_LC - intercept_LT) / (slope_LT - slope_LC)
    #     y_intersect_LC_LT_point = slope_LT * x_intersect_LC_LT_point + intercept_LT
    #
    #     # Расчет абсолютной погрешности
    #     absolute_difference = abs(slope_LT - slope_LC)
    #
    #     # Проверяем, являются ли оба угла нулевыми
    #     if slope_LT == slope_LC == 0:
    #         up_MDB.append(UpMDB(df, t1up, t2up, t3up, t4up, LT_up, LC_up))
    #     else:
    #         # Расчет среднего значения двух углов наклона
    #         average_slope = (slope_LT + slope_LC) / 2
    #
    #         # Расчет относительной погрешности
    #         relative_difference = (absolute_difference / average_slope) * 10
    #
    #         # Проверка, являются ли прямые параллельными (с некоторой погрешностью)
    #         # tolerance = 0.782122905029122190500000005  # установите допустимую погрешность в соответствии с вашими требованиями
    #         # if relative_difference < tolerance:
    #         print('absolute_difference', absolute_difference)
    #         tolerance = 0.95
    #         if absolute_difference < tolerance:
    #             up_MDB.append(UpMDB(df, t1up, t2up, t3up, t4up, LT_up, LC_up))
    #
    #     # Проверяем, где точка пересечения находится относительно t1up и t4up
    #     if x_intersect_LC_LT_point <= t1up[0]:
    #         CP_up_point = (x_intersect_LC_LT_point, y_intersect_LC_LT_point)
    #         up_EXP.append(UpEXP(df, t1up, t2up, t3up, t4up, LT_up, LC_up, CP_up_point))
    #
    #
    #     elif x_intersect_LC_LT_point >= t4up[0]:
    #         HP_up_point = (x_intersect_LC_LT_point, y_intersect_LC_LT_point)
    #         up_ATR.append(UpATR(df, t1up, t2up, t3up, t4up, LT_up, LC_up, HP_up_point))
    #
    #     print(f'Количество найденных комбинаций close/close: {len(up_EXP)}')
    #
    # return up_ATR, up_MDB, up_EXP
