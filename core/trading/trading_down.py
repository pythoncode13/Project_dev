import logging

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler

# функция для вычисления длины между двумя точками
def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def risk_reward_ratio_short(entry_price, take_price, stop_price):

    potential_risk = abs(stop_price - entry_price)
    potential_reward = abs(entry_price - take_price)

    if potential_risk != 0:  # Проверка на ноль, чтобы избежать ошибки деления на ноль
        risk_reward_ratio = potential_reward / potential_risk
    else:
        risk_reward_ratio = float('inf')  # Бесконечность
    return risk_reward_ratio

def take_pice_up(t1down, t2down):

    m = (t2down[1] - t1down[1]) / (t2down[0] - t1down[0])
    b = t1down[1] - m * t1down[0]

    vline_x = t2down[0] + 2 * (t2down[0] - t1down[0])
    x_intersect = vline_x
    y_intersect_down_take = m * x_intersect + b

    down_take_lines = (x_intersect, y_intersect_down_take)

    return down_take_lines

def take_price_down_t2_t3(t2down, t3down):

    m = (t3down[1] - t2down[1]) / (t3down[0] - t2down[0])
    b = t2down[1] - m * t2down[0]

    vline_x = t2down[0] - 2 * (t3down[0] - t2down[0])
    x_intersect = vline_x
    y_intersect_down_take = m * x_intersect + b

    down_take_lines = (x_intersect, y_intersect_down_take)

    return down_take_lines

def take_pice_down_x1(t1down, t2down):
    m = (t2down[1] - t1down[1]) / (t2down[0] - t1down[0])
    b = t1down[1] - m * t1down[0]

    vline_x = t2down[0] + 1 * (t2down[0] - t1down[0])
    x_intersect = vline_x
    y_intersect_down_take = m * x_intersect + b

    down_take_lines_x1 = (x_intersect, y_intersect_down_take)

    return down_take_lines_x1

def calculate_param_t2_t3_down_enter(t2down, t3down, down_enter_point):

    # Предположим, что точки определены следующим образом:
    t2down = np.array(t2down)
    t3down = np.array(t3down)
    down_enter_point = np.array(down_enter_point[0])

    # Угол наклона между t3down и down_enter_point
    angle_t3_enter = math.atan2(down_enter_point[1] - t3down[1], down_enter_point[0] - t3down[0])

    # Радиус кривизны между t2down, t3down и down_enter_point
    # Используем формулу радиуса кривизны для трех точек
    radius_curvature_t2_t3_enter = abs((t3down[1] - t2down[1]) * (down_enter_point[0] - t3down[0]) - (t3down[0] - t2down[0]) * (
                down_enter_point[1] - t3down[1])) / np.sqrt(
        (t3down[0] - t2down[0]) ** 2 + (t3down[1] - t2down[1]) ** 2) ** 3

    return angle_t3_enter, radius_curvature_t2_t3_enter

def take_line_plot(down_take_lines, dist_t1_t3_bar, t1down):
    plt.axvline(x=down_take_lines[0], color='g', linestyle='--', linewidth=0.5)

    # Конечная точка линии
    end_point = down_take_lines[0] + dist_t1_t3_bar * 3
    # Рисуем линию
    plt.plot([down_take_lines[0], end_point], [down_take_lines[1], down_take_lines[1]], color='g',
             linestyle='--', linewidth=0.5)

    # Добавляем текст
    plt.text(down_take_lines[0], down_take_lines[1], 'TAKE LINE', color='g', fontsize=8, va='bottom')

    plt.text(t1down[0], t1down[1], 't1down', color='black',
             rotation='vertical', va='top')

def compute_line(point1, point2, resolution=300):
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
    x_intersect = (intercept2 - intercept1) / (slope1 - slope2)
    y_intersect = slope1 * x_intersect + intercept1
    return (x_intersect, y_intersect)

def compute_tail_to_body_ratio(df, index, tail_type='lower'):
    row = df.loc[index]

    # Вычисляем тело свечи
    body = abs(row['open'] - row['close'])

    # Вычисляем хвост свечи
    if tail_type == 'lower':
        tail = row['open'] - row['low'] if row['open'] > row['close'] else row['close'] - row['low']
    elif tail_type == 'upper':
        tail = row['high'] - row['open'] if row['open'] < row['close'] else row['high'] - row['close']
    else:
        raise ValueError(f"Unknown tail_type: {tail_type}. Choose between 'lower' or 'upper'.")

    # Вычисляем соотношение хвоста и тела свечи
    tail_to_body_ratio = tail / body if body != 0 else 0

    return tail_to_body_ratio


def get_max_distance_point(df, Line_t1_t2, slope_Line_t1_t2, intercept_Line_t1_t2, t1down):
    # Инициализируем самое большое расстояние нулем
    max_distance = 0
    # И индекс бара, находящегося на самом большом расстоянии, None
    max_distance_bar_index = None
    max_distance_bar_price = None

    # Проходим по всем точкам на линии Line_t1_t2
    for x_value, y_value in zip(Line_t1_t2[0], Line_t1_t2[1]):
        # Находим бар, который ближе всего по времени к x_value
        nearest_bar_index = int(round(x_value))
        if 0 <= nearest_bar_index < len(df):
            bar = df.iloc[nearest_bar_index]
        else:
            # Здесь вы можете обработать случай, когда индекс выходит за пределы
            logging.debug(f"Index {nearest_bar_index} is out of bounds")

        bar_low = bar['low']

        # Вычисляем расстояние от low этого бара до точки на линии
        distance_bars = abs(bar_low - y_value)

        # Если это расстояние больше текущего максимального
        if distance_bars > max_distance:
            # обновляем максимальное расстояние и индекс бара
            max_distance = distance_bars
            max_distance_bar_index = nearest_bar_index
            max_distance_bar_price = bar_low

    point_t1_t2 = (max_distance_bar_index, max_distance_bar_price)

    x_t1_t2, y_t1_t2 = point_t1_t2
    y_line_at_x_t1_t2 = slope_Line_t1_t2 * x_t1_t2 + intercept_Line_t1_t2

    if y_t1_t2 >= y_line_at_x_t1_t2:
        point_t1_t2 = (t1down[0], df.loc[t1down[0], 'low'])
        # Вычисляем y для x = t1down_x на линии Line_t1_t2
        y_line_at_t1down_x = slope_Line_t1_t2 * point_t1_t2[0] + intercept_Line_t1_t2
        # Вычисляем расстояние между t1down и линией Line_t1_t2
        distance_t1down_line = abs(point_t1_t2[1] - y_line_at_t1down_x)
        max_distance = distance_t1down_line

    logging.debug('max_distance', max_distance)

    plt.plot(point_t1_t2[0], point_t1_t2[1], color='red',
             marker='^',
             markersize=10, markeredgecolor='black')
    # max_distance_bar_index теперь содержит индекс бара, находящегося на самом большом расстоянии от линии Line_t1_t2

    return max_distance, point_t1_t2

def compute_stats(*points):
    y_values = [point[1] for point in points]
    mean_y = np.mean(y_values)
    std_dev_y = np.std(y_values)
    std_dev_y_mean_y = std_dev_y / mean_y
    return mean_y, std_dev_y, std_dev_y_mean_y

def calculate_statistics(df, t1, t2, t3):
    def slice_df(df, start, end, column):
        start = max(0, start)
        return df.loc[start:end, column]

    def percentage_difference(a, b):
        return ((a - b) / a) * 100

    df_slice_low = slice_df(df, t1[0] - 15, t1[0], 'low')
    df_slice_close = slice_df(df, t1[0] - 15, t1[0], 'close')

    min_low = df_slice_low.min()
    min_close = df_slice_close.min()
    max_close = df_slice_close.max()

    percentage_difference_min_low_and_t1 = percentage_difference(t1[1], min_low)
    percentage_difference_min_close_and_t1 = percentage_difference(t1[1], min_close)
    percentage_difference_max_close_and_t1 = percentage_difference(t1[1], max_close)

    price_change_t1_t2 = percentage_difference(t2[1], t1[1])
    price_change_t2_t3 = percentage_difference(t2[1], t3[1])
    diff_price_change = abs(price_change_t1_t2 - price_change_t2_t3)

    return (percentage_difference_min_low_and_t1,
            percentage_difference_min_close_and_t1,
            percentage_difference_max_close_and_t1,
            price_change_t1_t2,
            price_change_t2_t3,
            diff_price_change
            )

def compute_all_stats(t1down, t2down, t3down, t4down, intersection_t1_point_t2_t3_lines, point_t1_t2):
    mean_y, std_dev_y, std_dev_y_mean_y = compute_stats(t1down, t2down, t3down)
    mean_y_1, std_dev_y_1, std_dev_y_mean_y_1 = compute_stats(t1down, t2down, t3down, t4down)
    mean_y_2, std_dev_y_2, std_dev_y_mean_y_2 = compute_stats(t1down, intersection_t1_point_t2_t3_lines, point_t1_t2)
    mean_y_3, std_dev_y_3, std_dev_y_mean_y_3 = compute_stats(t1down, t2down, t3down, t4down, intersection_t1_point_t2_t3_lines, point_t1_t2)
    mean_y_4, std_dev_y_4, std_dev_y_mean_y_4 = compute_stats(t1down, t2down, t3down)
    mean_y_5, std_dev_y_5, std_dev_y_mean_y_5 = compute_stats(t1down, t2down, t4down)
    mean_y_6, std_dev_y_6, std_dev_y_mean_y_6 = compute_stats(t1down, t3down, intersection_t1_point_t2_t3_lines)
    mean_y_7, std_dev_y_7, std_dev_y_mean_y_7 = compute_stats(t2down, t4down, point_t1_t2)
    _, std_dev_y_t2_t3_up_enter, _ = compute_stats(t2down, t3down, t4down)

    return std_dev_y_mean_y, std_dev_y_mean_y_1, std_dev_y_mean_y_2, std_dev_y_mean_y_3, std_dev_y_mean_y_4, std_dev_y_mean_y_5, std_dev_y_mean_y_6, std_dev_y_mean_y_7, std_dev_y_t2_t3_up_enter

def draw_marker(point, color='red', marker='^', markersize=10, markeredgecolor='black'):
    plt.plot(point[0], point[1], color=color,
             marker=marker,
             markersize=markersize, markeredgecolor=markeredgecolor)


def process_group_short(df, down_trend_points):

    down_enter_points = []
    down_stop_points = []
    down_take_points = []

    all_other_parametrs = []

    for points in down_trend_points:
        t1down = points[0]
        t2down = points[1]
        t3down = points[2]

        down_stop_point = None
        down_take_point = None

        for j in range(t3down[0], len(df)):

            if t1down[0]+1 == t2down[0] or any(df.loc[t1down[0]+1:t3down[0], 'high'] >= t1down[1]) \
                    or any(df.loc[t2down[0]+1:t3down[0], 'low'] <= t2down[1])\
                    or any(df.loc[t1down[0]+1:t2down[0]-1, 'low'] == t2down[1]):
                continue

            if (df.loc[j, 'low']) < t2down[1] and j != t3down[0]:

                entry_price = t2down[1]

                stop_price = t1down[1] - 0.5 * (t1down[1] - t2down[1])

                down_take_lines = take_pice_up(t1down, t2down)
                take_price = t2down[1] - 0.9 * (t2down[1] - down_take_lines[1])

                risk_reward_ratio = risk_reward_ratio_short(entry_price, take_price, stop_price)
                percent_difference = ((entry_price - take_price) / entry_price) * 100 # разница для шорт позиций

                if abs(percent_difference) > 0.4 and not any(df.loc[t3down[0]:j, 'low'] <= take_price) and not any(df.loc[t3down[0]+1:j, 'high'] >= t3down[1]) and 1.0 <= risk_reward_ratio:

                    entry_date = df.loc[j, 'dateTime']

                    entry_index = j
                    down_enter_point = ((entry_index, entry_price), stop_price, take_price,
                                        entry_date)

                    '''Вводим дополнительные точки'''

                    angle_deg_LT, slope_LT, intercept_LT, LT_down = compute_line(t1down, t3down)
                    angle_deg_LC, slope_LC, intercept_LC, LC_down = compute_line(t2down, down_enter_point[0])
                    angle_deg_Line_t1_t2, slope_Line_t1_t2, intercept_Line_t1_t2, Line_t1_t2 = compute_line(t1down, t2down)

                    # Находим точку максимального расстояния от Line_t1_t2
                    max_distance, point_t1_t2 = get_max_distance_point(df, Line_t1_t2, slope_Line_t1_t2, intercept_Line_t1_t2, t1down)
                    draw_marker(point_t1_t2, color='red', marker='^', markersize=10, markeredgecolor='black')

                    angle_deg_Line_t1_point_t2_t3, slope_Line_t1_point_t2_t3, intercept_Line_t1_point_t2_t3, Line_t1_point_t2_t3 = compute_line(t3down,
                                                                                                            point_t1_t2)

                    # Находим СТ, точку пересечения линий т1:т2 и т1_поинт_т2:т3
                    if slope_LT != slope_LC:
                        CP_down_point = find_intersection(slope_LT, intercept_LT, slope_LC, intercept_LC)

                    intersection_t1_point_t2_t3_lines = find_intersection(slope_Line_t1_t2, intercept_Line_t1_t2,
                                                                          slope_Line_t1_point_t2_t3,
                                                                          intercept_Line_t1_point_t2_t3)

                    down_enter_points.append(down_enter_point)

                    stop_point_found = False
                    take_point_found = False
                    take_point_g = None
                    stop_point_g = None

                    for g in range(down_enter_point[0][0] + 1, len(df)):

                        if df.loc[g, 'high'] >= stop_price:
                            stop_point_found = True
                            stop_point_g = g
                            break  # Прерываем цикл, так как условие выполнено

                    for g in range(down_enter_point[0][0], len(df)):
                        if df.loc[g, 'low'] <= take_price:
                            take_point_found = True
                            take_point_g = g
                            break

                    if take_point_found and (not stop_point_found or take_point_g < stop_point_g):
                        take_date = df.loc[take_point_g, 'dateTime']
                        down_take_point = (t2down, (take_point_g, take_price), stop_price, take_price, take_date)
                        down_take_points.append(down_take_point)

                    elif stop_point_found and (not take_point_found or stop_point_g < take_point_g):
                        stop_date = df.loc[stop_point_g, 'dateTime']
                        down_stop_point = (t2down, (stop_point_g, stop_price), stop_price, take_price, stop_date)
                        down_stop_points.append(down_stop_point)
                        stop_point_found = True

                    else:
                        logging.debug("Ни одна из цен не была достигнута.")

                    # Инициализация переменной
                    close_position_point = None
                    # Проверка условий
                    if down_stop_point is not None:
                        close_position_point = down_stop_point
                    elif down_take_point is not None:
                        close_position_point = down_take_point
                    else:
                        close_position_point = None

                    diff = None
                    profit_or_lose = None
                    if close_position_point is not None:
                        diff = round(
                            ((down_enter_point[0][1] - close_position_point[1][1]) / down_enter_point[0][1] * 100), 5)
                        profit_or_lose = None
                        if diff is not None:
                            profit_or_lose = 1 if diff > 0 else 0

                            down_enter_point = ((entry_index, entry_price), stop_price, take_price, entry_date)
                            down_enter_points.append(down_enter_point)

                            color = 'green' if diff > 0 else 'yellow'
                            point = down_take_point if diff > 0 else down_stop_point
                            point_text = 'down_take_point' if diff > 0 else 'down_stop_point {:.2f}'.format(
                                down_stop_point[1][1])

                            plt.plot(point[1][0], point[1][1], color=color, marker='^', markersize=10,
                                     markeredgecolor='black')
                            plt.text(point[1][0], point[1][1], point_text, fontsize=7, color='b')

                            for _ in range(2):
                                plt.plot(down_enter_point[0][0], down_enter_point[0][1], color='red', marker='^',
                                         markersize=10, markeredgecolor='black')
                                plt.text(down_enter_point[0][0], down_enter_point[0][1],
                                         'down_enter_point {:.2f}'.format(down_enter_point[0][1]), fontsize=7,
                                         color='b')
                                plt.text(down_enter_point[0][0], down_enter_point[0][1], entry_date, color='black',
                                         rotation='vertical', va='top')

                                #  Рисуем ЛЦ_довн
                                plt.plot(LC_down[0], LC_down[1], ':', color='purple', linewidth=0.9)
                                # Продлеваем ЛТ_довн и ЛЦ_довн для нахождения СТ
                                plt.plot([CP_down_point[0], t1down[0]], [CP_down_point[1], t1down[1]], ':',
                                         color='purple',
                                         linewidth=0.9)
                                plt.scatter(CP_down_point[0], CP_down_point[1], s=80, facecolors='none', edgecolors='b')

                                # Линия Line_t1_point_t2_t3
                                plt.plot(Line_t1_t2[0], Line_t1_t2[1], ':', color='purple', linewidth=1.9)
                                # Линия Line_t1_point_t2_t3
                                plt.plot(Line_t1_point_t2_t3[0], Line_t1_point_t2_t3[1], ':', color='purple', linewidth=1.9)
                                # Точка пересечения двух линий
                                draw_marker(intersection_t1_point_t2_t3_lines, color='yellow', marker='x',
                                            markersize=10,
                                            markeredgecolor='black')
                                # Количество баров т1-т3
                                dist_t1_t3_bar = t3down[0] - t1down[0]
                                take_line_plot(down_take_lines, dist_t1_t3_bar, t1down)

                    close_position_point_price = None
                    if close_position_point is not None:
                        close_position_point_price = close_position_point[1][1]



                    # Создаем список точек
                    down_enter_point_pre_norm = (down_enter_point[0][0], down_enter_point[0][1])
                    points_i = [CP_down_point, t1down, t2down, t3down, down_enter_point_pre_norm, point_t1_t2, intersection_t1_point_t2_t3_lines]
                    # Преобразуем список в массив Numpy
                    points_array = np.array(points_i)
                    # Инициализируем MinMaxScaler
                    scaler = MinMaxScaler()
                    # Применяем MinMaxScaler к массиву точек
                    points_norm = scaler.fit_transform(points_array)
                    # Получаем нормализованные координаты точек
                    CP_down_point_norm, t1down_norm, t2down_norm, t3down_norm, down_enter_point_pre_norm, point_t1_t2_norm, intersection_t1_point_t2_t3_lines_norm = points_norm

                    # Расчет длин
                    length3_point_t1_t2_norm = distance(t1down, intersection_t1_point_t2_t3_lines_norm)

                    # Угол наклона между t2down и t3down
                    angle_t2_t3 = math.atan2(t3down[1] - t2down[1], t3down[0] - t2down[0])
                    # Площадь под кривой между t1down и t3down
                    # Используем метод трапеций для приближения интеграла
                    area_under_curve = 0.5 * (t3down[0] - t1down[0]) * (t1down[1] + t3down[1])
                    angle_t3_enter, radius_curvature_t2_t3_enter = calculate_param_t2_t3_down_enter(
                        t2down, t3down, down_enter_point)

                    percentage_difference_min_low_and_t1, percentage_difference_min_close_and_t1, percentage_difference_max_close_and_t1, price_change_t1_t2, price_change_t2_t3, diff_price_change = calculate_statistics(
                        df, t1down, t2down, t3down_norm)

                    # Относительный процент от точки входа в тейку
                    price_change_t1_t2 = (t1down[1] - t2down[1]) / (t2down[1])
                    price_change_t2_t3 = (t3down_norm[1] - t2down[1]) / t2down[1]
                    diff_price_change = abs(price_change_t1_t2 - price_change_t2_t3)

                    # std_dev_y_mean_y
                    # Среднее значение y-координаты  и стандартное отклонение y-координаты между точками
                    t4down = down_enter_point[0]
                    std_dev_y_mean_y, std_dev_y_mean_y_1, std_dev_y_mean_y_2, std_dev_y_mean_y_3, std_dev_y_mean_y_4, std_dev_y_mean_y_5, std_dev_y_mean_y_6, std_dev_y_mean_y_7, std_dev_y_t2_t3_up_enter = compute_all_stats(
                        t1down, t2down, t3down, t4down, intersection_t1_point_t2_t3_lines, point_t1_t2)

                    # Вычисляем соотношение хвоста и тела свечи для различных точек
                    tail_to_body_ratio_t1 = compute_tail_to_body_ratio(df, t1down[0], tail_type='upper')
                    tail_to_body_ratio_t2 = compute_tail_to_body_ratio(df, t2down[0])
                    tail_to_body_ratio_t3 = compute_tail_to_body_ratio(df, t3down[0], tail_type='upper')
                    tail_to_body_ratio_enter_point_back_1 = compute_tail_to_body_ratio(df, down_enter_point[0][0] - 1,
                                                                                       tail_type='upper')

                    # indicators
                    rsi_value1 = df.loc[t1down[0], 'rsi']
                    rsi_value2 = df.loc[t2down[0], 'rsi']
                    rsi_value3 = df.loc[t3down[0], 'rsi']
                    rsi_value_enter = df.loc[down_enter_point[0][0], 'rsi']

                    vwap_t1 = df.loc[t1down[0], 'vwap']
                    vwap_enter = df.loc[down_enter_point[0][0], 'vwap']
                    vwap_ratio_t1 = t1down[0] / vwap_t1
                    vwap_ratio_enter = down_enter_point[0][1] / vwap_enter


                    other_parametrs = (
                        entry_date,
                        entry_price,
                        stop_price, take_price,
                        close_position_point_price, diff, profit_or_lose, risk_reward_ratio,
                        # percentage
                        diff_price_change,
                        percentage_difference_min_low_and_t1,
                        percentage_difference_min_close_and_t1,
                        percentage_difference_max_close_and_t1,
                        # std_dev_y_mean_y
                        std_dev_y_mean_y,
                        std_dev_y_mean_y_1,
                        std_dev_y_mean_y_2,
                        std_dev_y_mean_y_3,
                        std_dev_y_mean_y_4,
                        std_dev_y_mean_y_5,
                        std_dev_y_mean_y_6,
                        std_dev_y_mean_y_7,
                        std_dev_y_t2_t3_up_enter,
                        # bar ratio
                        tail_to_body_ratio_t1,
                        tail_to_body_ratio_t2,
                        tail_to_body_ratio_t3,
                        tail_to_body_ratio_enter_point_back_1,
                        # geometric
                        area_under_curve,
                        radius_curvature_t2_t3_enter,
                        angle_t2_t3,
                        angle_t3_enter,
                        angle_deg_LT,
                        max_distance,
                        length3_point_t1_t2_norm,
                        # indicators
                        rsi_value1,
                        rsi_value2,
                        rsi_value3,
                        rsi_value_enter,
                        vwap_ratio_t1,
                        vwap_ratio_enter
                    )

                    all_other_parametrs.append(other_parametrs)

                    break
                break
    return all_other_parametrs
