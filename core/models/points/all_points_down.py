import pandas as pd
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from core.models.down_exp_property import Line, Point
from core.models.downmodelclass import DownModel


def find_t1_down(df: pd.DataFrame) -> List[Tuple[int, float]]:

    all_t1_down = []

    df['high_max'] = df['high'].rolling(11, center=True).max()
    df['high_max_prev_5'] = df['high'].shift(1).rolling(
        5).min()
    for i in range(5, len(df) - 5):
        curr_candle = df.iloc[i]

        if curr_candle['high'] == curr_candle['high_max']:
            all_t1_down.append((i, curr_candle['high']))
    return all_t1_down


def find_t2_down(df: pd.DataFrame) -> List[Tuple[int, float]]:
    # Поиск t2
    all_t2_down = []

    for i in range(2, len(df) - 2):
        curr_candle = df.iloc[i]
        next_candle = df.iloc[i + 1]

        if curr_candle['low'] <= next_candle['low']:
            # Проверка на наличие предыдущей t2 с тем же значением 'low'. Если такая есть, пропускаем эту свечу.
            if all_t2_down and all_t2_down[-1][1] == curr_candle['low']:
                continue
            all_t2_down.append((i, curr_candle['low']))

    return all_t2_down


def find_t3_down(df: pd.DataFrame) -> List[Tuple[int, float]]:
    # Поиск t3
    all_t3_down = []

    for i in range(2, len(df) - 2):
        curr_candle = df.iloc[i]
        next_candle = df.iloc[i + 1]  # получаем следующую свечу
        prev_candles = df.iloc[i - 1]

        if curr_candle['high'] >= next_candle['high'] and curr_candle['high'] >= prev_candles['high']:
            # Проверка на наличие предыдущей t2 с тем же значением 'high'. Если такая есть, заменяем её этой свечой.
            if all_t3_down and all_t3_down[-1][1] == curr_candle['high']:
                all_t3_down[-1] = (i, curr_candle['high'])
            else:
                all_t3_down.append((i, curr_candle['high']))

    return all_t3_down


def find_t4_down(df: pd.DataFrame) -> List[Tuple[int, float]]:
    """Поиск t4"""
    all_t4_down = []
    for i in range(3, len(df) - 4):
        curr_candle = df.iloc[i]
        prev_candles = df.iloc[i - 3: i]
        next_candle = df.iloc[i + 1]
        next_next_candle = df.iloc[i + 2]

        if (curr_candle['low'] < prev_candles['low'].max()
            and curr_candle['low'] <= next_candle['low']):
            if curr_candle['high'] < next_candle['high'] or ((
                    next_candle['high'] < next_next_candle['high'] and
                    next_candle['low'] < next_next_candle['low']) or (
                    curr_candle['high'] < next_next_candle['high'])
            ):
                all_t4_down.append((i, curr_candle['low']))

    return all_t4_down


def get_combinations(df, range_limit=30):
    """
    Находит все комбинации из t1, t2, t3.

    Parameters
    ----------
    df : pandas.DataFrame
        Оригинальный DataFrame.
    all_t1_down : List[Tuple[int, float]]
        Список точек t1.
    all_t2_down : List[Tuple[int, float]]
        Список точек t2.
    all_t3_down : List[Tuple[int, float]]
        Список точек t3.
    range_limit : int, optional
        Диапазон, в котором ищем t4. По умолчанию 30.

    Returns
    -------
    List[Tuple[Tuple[int, float], Tuple[int, float], Tuple[int, float]]]
        Список комбинаций.
    """

    # Ищем точки-экстремумы по hl свечей
    all_t1_down = find_t1_down(df)
    all_t2_down = find_t2_down(df)
    all_t3_down = find_t3_down(df)

    # Создаем DataFrame для удобства работы
    df_t1 = pd.DataFrame(all_t1_down, columns=['idx', 'price'])
    df_t2 = pd.DataFrame(all_t2_down, columns=['idx', 'price'])
    df_t3 = pd.DataFrame(all_t3_down, columns=['idx', 'price'])

    # Создаем пустой список для хранения результатов
    combinations = []

    # Создаем массивы для удобства работы с NumPy
    t1_array = df_t1.to_numpy()
    t2_array = df_t2.to_numpy()
    t3_array = df_t3.to_numpy()

    for t1 in t1_array:
        # Вычисляем пределы для t3
        start = t1[0]
        end = start + range_limit

        # Выбираем точки t3, которые попадают в диапазон и у которых цена выше чем у t1
        t3_filtered = t3_array[
            (t3_array[:, 0] >= start) & (t3_array[:, 0] <= end) & (
                        t1[1] >= t3_array[:, 1])]

        for t3 in t3_filtered:

            # Выбираем точки t2, которые находятся между t1 и t3
            t2_filtered = t2_array[(t2_array[:, 0] > t1[0]) & (t2_array[:, 0] < t3[0])]

            # Если t2_filtered пуст, пропускаем эту комбинацию
            if t2_filtered.size == 0:
                continue

            # Выбираем точку t2 с наименьшим значением 'price'
            t2 = t2_filtered[np.argmin(t2_filtered[:, 1])]

            if t1[0] == t2[0]:
                continue

            if t2[0] == t3[0]:
                continue

            if t2[1] != min(df.loc[t1[0]+1:t3[0]]['low']):
                continue

            # Вычисляем угловой коэффициент прямой между t1 и t3
            slope = (t3[1] - t1[1]) / (t3[0] - t1[0])

            # Вычисляем значения прямой для всех точек между t1 и t3
            line_values = t1[1] + slope * (np.arange(t1[0] + 1, t3[0] + 1) - t1[0])  # add 1 bars to t3

            # Находим максимальную цену в диапазоне t1:t3
            max_price = df.loc[t1[0]+1:t3[0], 'high'].values  # add 3 bars to t3

            # Если хотя бы одна цена меньше соответствующего значения прямой, пропускаем эту комбинацию
            if np.any(max_price > line_values):
                # Вычисляем угловой коэффициент прямой между t1 и t3
                t1_next = (t1[0]+1, df.loc[t1[0]+1, 'high'])
                t1 = t1_next
                # Вычисляем угловой коэффициент прямой между t1 и t3
                slope = (t3[1] - t1[1]) / (t3[0] - t1[0])

                # Вычисляем значения прямой для всех точек между t1 и t3
                line_values = t1[1] + slope * (
                            np.arange(t1[0] + 1, t3[0] + 1) - t1[
                        0])  # add 1 bars to t3

                # Находим максимальную цену в диапазоне t1:t3
                max_price = df.loc[t1[0] + 1:t3[0],
                            'high'].values  # add 3 bars to t3
                if np.any(max_price > line_values):
                    continue

            # Иначе добавляем комбинацию в список
            combinations.append((t1, t2, t3))

    return combinations


def add_t4_to_combinations(df, range_limit=30):
    """
    Добавляет t4 к комбинациям, если они удовлетворяют условиям.

    Parameters
    ----------
    df : pandas.DataFrame
        Оригинальный DataFrame.
    combinations : List[Tuple[Tuple[int, float], Tuple[int, float], Tuple[int, float]]]
        Список комбинаций.
    all_t4_down : List[Tuple[int, float]]
        Список точек t4.
    range_limit : int, optional
        Диапазон, в котором ищем t4. По умолчанию 30.

    Returns
    -------
    List[Tuple[Tuple[int, float], Tuple[int, float], Tuple[int, float], Tuple[int, float]]]
        Список комбинаций с добавленными t4.
    """
    combinations = get_combinations(df, range_limit=30)

    all_t4_down = find_t4_down(df)

    # Создаем DataFrame для удобства работы
    df_t4 = pd.DataFrame(all_t4_down, columns=['idx', 'price'])

    # Создаем новый список для хранения обновленных комбинаций
    new_combinations = []

    # Создаем массив для удобства работы с NumPy
    t4_array = df_t4.to_numpy()

    for combination in combinations:
        t1, t2, t3 = combination

        # Вычисляем пределы для t4
        start = t3[0]
        end = start + range_limit

        # Выбираем точки t4, которые попадают в диапазон и у которых цена ниже чем у t3 и t2
        t4_filtered = t4_array[(t4_array[:, 0] > start) & (t4_array[:, 0] <= end) & (t2[1] > t4_array[:, 1])]

        for t4 in t4_filtered:

            # Вычисляем угловой коэффициент прямой между t1 и t3
            slope = (t3[1] - t1[1]) / (t3[0] - t1[0])

            # Вычисляем значения прямой для всех точек между t1 и t4
            line_values = t3[1] + slope * (np.arange(t3[0] + 3, t4[0] + 1) - t3[0])  # add 1 bar to t4

            # Находим минимальную цену в диапазоне t1:t4
            max_price = df.loc[t3[0]+3:t4[0], 'high'].values  # add 1 bar to t4

            # Если хотя бы одна цена меньше соответствующего значения прямой, пропускаем эту комбинацию
            if np.any(max_price > line_values):
                continue

            # Вычисляем угловой коэффициент прямой между t1 и t3
            slope = (t4[1] - t2[1]) / (t4[0] - t2[0])

            # Вычисляем значения прямой для всех точек между t1 и t4
            line_values = t2[1] + slope * (
                        np.arange(t2[0] + 1, t4[0] - 1) - t2[
                    0])  # add 1 bar to t4

            # Находим минимальную цену в диапазоне t1:t4
            min_price = df.loc[t2[0] + 1:t4[0] - 2,
                        'low'].values  # add 1 bar to t4

            # Если хотя бы одна цена меньше соответствующего значения прямой, пропускаем эту комбинацию
            if np.any(min_price < line_values):
                continue
            if t4[1] != min(df.loc[t2[0]+1:t4[0]]['low']):
                continue

            if t1[0] == t2[0]:
                continue

            # Иначе добавляем комбинацию в список
            new_combinations.append((t1, t2, t3, t4))

    return new_combinations


def find_start_points_hl_down(df: pd.DataFrame):

    new_combinations = add_t4_to_combinations(df, range_limit=30)

    down_trend_points = []
    down_trend_points_test = []
    for combination in new_combinations:
        t1down = combination[0]
        t2down = combination[1]
        t3down = combination[2]
        t4down = combination[3]

        if t3down[1] == max(df.loc[t1down[0] + 1:t3down[0]]['low']):
            continue

        # Проверяем, что т3 это мин лоу на участке т3-т4
        high_below_t4down = df.loc[t3down[0]:t4down[0], 'high'].min() > t3down[1]
        if high_below_t4down:
            continue

        # проводим линию ЛТ
        slope_LT, intercept_LT, LT_down = Line.calculate(t1down, t3down)
        # валидация
        if Line.check_line(df, slope_LT, intercept_LT, t1down, t4down,
                           direction='high'):
            t3down1 = Line.correction_LT(df, t3down, t4down, slope_LT, intercept_LT,
                          return_rightmost=True)
            slope_LT, intercept_LT, LT_down = Line.calculate(t1down, t3down1)

        # проводим линию ЛЦ
        slope_LC, intercept_LC, LC_down = Line.calculate(t2down, t4down)

        # валидация
        if Line.check_line(df, slope_LC, intercept_LC, (t1down[0]+1, 0), t4down,
                           direction='low'):
            t4down1 = Line.correction_LC_t4down1(df, t2down, t4down, slope_LC, intercept_LC)
            t2down1 = Line.correction_LC_t2down1(df, t1down, t2down, t4down1)

            slope_LC, intercept_LC, LC_down = Line.calculate(t2down1, t4down1)
            if Line.check_line(df, slope_LC, intercept_LC, (t1down[0]+1, 0), t4down1,
                               direction='low'):
                continue

        # Поиск точки пересечения прямых ЛТ и ЛЦ
        x_intersect_LC_LT_point = (intercept_LC - intercept_LT) / (slope_LT - slope_LC)
        y_intersect_LC_LT_point = slope_LT * x_intersect_LC_LT_point + intercept_LT
        CP_down_point = (x_intersect_LC_LT_point, y_intersect_LC_LT_point)

        if x_intersect_LC_LT_point >= t4down[0]:
            continue

        if slope_LT == slope_LC == 0:
            continue

        parallel = Line.cos_sim(slope_LT, slope_LC)
        plt.text(t4down[0], t4down[1], parallel, fontsize=10)

        if parallel > 60:
            continue

        # _, _, LC_line = Line.calculate(t2down, t4down)

        plt.plot(LC_down[0], LC_down[1], ':', color='purple',
                 linewidth=0.9)

        plt.plot(t2down[0], t2down[1], 'o', color='y')

        plt.plot(t4down[0], t4down[1], 'o', color='k')
        # plt.plot(t1down[0], t1down[1], 'o', color='k')

        plt.plot(t3down[0], t3down[1], 'o', color='r')

        dist_cp_t4_x2 = t4down[0] + ((t4down[0] - x_intersect_LC_LT_point) * 2)

        upper_limit = min(int(dist_cp_t4_x2), len(df))

        # Находим бар, который пробил уровень т4
        first_bar_above_t4down = None
        t5down = None
        HP_down_point = None
        for i in range(int(t4down[0]), upper_limit):
            if df.loc[i, 'low'] < t4down[1]:
                first_bar_above_t4down = i
                break

        LT_break_point = Point.find_LT_break_point(df, t4down, upper_limit, slope_LT, intercept_LT)

        LT_break_point_close = None
        if LT_break_point:
            LT_break_point_close = Point.find_LT_break_point_close(df, t4down, upper_limit,
                                                       slope_LT, intercept_LT)

        if (
                # Если first_bar_above_t4down существует
                # и LT_break_point не существует
                not LT_break_point and first_bar_above_t4down
                # ИЛИ
                or
                # Если first_bar_above_t4down и LT_break_point оба существуют
                # и индекс first_bar_above_t4down меньше индекса
                # в кортеже LT_break_point
                (
                        LT_break_point and first_bar_above_t4down and first_bar_above_t4down <
                        LT_break_point[0])
        ):

            t5down_index = df.loc[(t4down[0] + 1):first_bar_above_t4down,
                         'high'].idxmax()
            t5down_price = df.loc[t5down_index, 'high']
            if t5down_price <= t4down[1]:
                break
            t5down = (t5down_index, t5down_price)

            # Проверка пересечение тел свечей т2-т5
            t2down_candle = df.loc[t2down[0]]
            t5down_candle = df.loc[t5down[0]]

            t2down_downper_body_edge = max(t2down_candle['open'],
                                       t2down_candle['close'])
            t5down_lower_body_edge = min(t5down_candle['open'],
                                       t5down_candle['close'])

            if t2down_downper_body_edge < t5down_lower_body_edge:
                continue
            else:
                # find HP
                # проводим линию HP
                slope_LT_HP, intercept_LT_HP, LT_down_HP = Line.calculate(t3down, t5down)
                # валидация
                if Line.check_line(df, slope_LT_HP, intercept_LT_HP, t3down, t5down,
                                   direction='high'):
                    t3down1 = Line.correction_LT_HP(df, t3down, t5down, slope_LT_HP,
                                               intercept_LT_HP)
                    slope_LT_HP, intercept_LT_HP, LT_down_HP = Line.calculate(t3down1, t5down)
                plt.plot(LT_down_HP[0], LT_down_HP[1], ':', color='purple',
                         linewidth=0.9)

                # Поиск точки пересечения прямых LT_down_HP и ЛЦ
                x_intersect_LC_LT_down_HP_point = (intercept_LC - intercept_LT_HP) / (
                            slope_LT_HP - slope_LC)
                y_intersect_LC_LT_down_HP_point = slope_LT_HP * x_intersect_LC_LT_down_HP_point + intercept_LT_HP

                if y_intersect_LC_LT_down_HP_point < t4down[1] + (t4down[1] - t1down[1]) * 5:
                    continue
                if x_intersect_LC_LT_down_HP_point < t4down[0]:
                    HP_down_point = None
                else:
                    HP_down_point = (x_intersect_LC_LT_down_HP_point, y_intersect_LC_LT_down_HP_point)

                    plt.hlines(
                        y=y_intersect_LC_LT_down_HP_point,
                        xmin=t4down[0] - 50,
                        xmax=t4down[0] + 50,
                        colors='g',
                        linestyles=':',
                        linewidth=1.5,
                    )


                    plt.plot(HP_down_point[0], HP_down_point[1], color='red',
                                 marker='>',
                                 markersize=3, markeredgecolor='black')

        LC_break_point = None
        if first_bar_above_t4down:
            LC_break_point = Point.find_LC_break_point(df, t4down, dist_cp_t4_x2, slope_LC,
                                                       intercept_LC)

        point_under_LC = None
        if LC_break_point:
            point_under_LC = Point.find_LT_break_point(df, LC_break_point, dist_cp_t4_x2, slope_LC, intercept_LC)

        plt.text(t2down[0], t2down[1], t2down[0], fontsize=10)

        up_take_4_100 = t4down[1] + (t4down[1] - t1down[1])
        plt.hlines(
            y=up_take_4_100,
            xmin=t4down[0],
            xmax=t4down[0] + 50,
            colors='r',
            linestyles='solid',
            linewidth=1.5,
        )

        # Вычисляем коэффициенты уравнения прямой
        m = (t2down[1] - t1down[1]) / (t2down[0] - t1down[0])
        b = t1down[1] - m * t1down[0]

        # Расширяем линию тренда на две длины от t1down до t2down
        vline_x = t2down[0] + 1 * (t2down[0] - t1down[0])

        # Находим точку пересечения
        x_intersect = vline_x
        y_intersect_down_take = m * x_intersect + b

        up_take_lines = (x_intersect, y_intersect_down_take)

        plt.hlines(
                    y=up_take_lines[1],
                    xmin=up_take_lines[0],
                    xmax=up_take_lines[0] + 50,
                    colors='r',
                    linestyles='solid',
                    linewidth=0.5,
                )

        # Вычисляем коэффициенты уравнения прямой
        m = (t4down[1] - t1down[1]) / (t4down[0] - t1down[0])
        b = t1down[1] - m * t1down[0]

        # Расширяем линию тренда на две длины от t1down до t2down
        vline_x = t4down[0] + 1 * (t4down[0] - t1down[0])

        # Находим точку пересечения
        x_intersect = vline_x
        y_intersect_down_take = m * x_intersect + b

        up_take_100 = t2down[1] + (t2down[1] - t1down[1]) * 1
        up_take_lines1 = (x_intersect, up_take_100)

        plt.hlines(
            y=up_take_lines[1],
            xmin=up_take_lines[0],
            xmax=up_take_lines[0] + 50,
            colors='r',
            linestyles='solid',
            linewidth=0.5,
        )

        plt.hlines(
            y=up_take_lines1[1],
            xmin=up_take_lines1[0],
            xmax=up_take_lines1[0] + 50,
            colors='g',
            linestyles='solid',
            linewidth=0.5,
        )

        plt.plot(LT_down[0], LT_down[1], ':', color='purple',
                 linewidth=0.9)


        plt.plot(LC_down[0], LC_down[1], ':', color='purple',
                 linewidth=0.9)

        dist_cp_t4_x1 = t4down[0] + ((t4down[0] - CP_down_point[0]) * 1)
        down_trend_points.append(DownModel(t1down, t2down, t3down, t4down, t5down, first_bar_above_t4down, up_take_lines1, dist_cp_t4_x2, HP_down_point, LC_break_point, point_under_LC, LT_break_point, CP_down_point, dist_cp_t4_x1, LT_break_point_close, slope_LT, intercept_LT))
        down_trend_points_test.append((t1down, t2down, t3down, t4down, t5down))

    return down_trend_points, down_trend_points_test
