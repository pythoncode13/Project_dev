import matplotlib.pyplot as plt

from core.base_trade_analysis_long import BaseTradeAnalysis
from core.up_model_property import UpModelProperty
from core.advanced_trade_analysis_long import AdvancedTradeAnalysis
from core.upmodelclass import UpModel
from core.up_exp_property import Line, Point

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def calculate_risk_reward_ratio(entry_price, stop_price, take_price):
    """Считает соотношение риск/прибыль."""

    potential_risk = entry_price - stop_price
    potential_reward = take_price - entry_price

    if potential_risk != 0:  # Избегаем деления на ноль
       risk_reward_ratio = potential_reward / potential_risk
    else:
       risk_reward_ratio = float('inf')  # Бесконечность

    return risk_reward_ratio

def find_max_high_between_t4up_and_lt_up(df, t4up, LT_up_breakout_point):
    max_high = None
    max_high_index = None
    passed_t4up = False
    high_below_LT_up = False

    for index, row in df.iterrows():
        if not passed_t4up:
            if index == t4up[0]:
                passed_t4up = True
                max_high = row['high']
                max_high_index = index
            continue

        high = row['high']
        close = row['close']
        # y_LT_down = LT_down[1][np.abs(np.array(LT_down[0]) - index).argmin()]

        if close < LT_up_breakout_point[1]:
            break

        if high > max_high:
            max_high = high
            max_high_index = index
    return (max_high_index, max_high)

def find_target_up(df, t1up, t4up, LT_up_breakout_point, CP_up_point):
    '''TARGET'''

    up_tg1 = []
    up_tg2 = []
    up_tg3 = []
    up_tg5 = []
    max_high = []

    if LT_up_breakout_point is not None:

        max_high_index, max_high = find_max_high_between_t4up_and_lt_up(df, t4up, LT_up_breakout_point)
        up_tg1 = LT_up_breakout_point[1] - (max_high - LT_up_breakout_point[1]) * 1.0

        up_tg2 = t1up[1]

        up_tg3 = LT_up_breakout_point[1] - (t4up[1] - t1up[1]) * 1.0

        up_tg5 = CP_up_point[1] - (max_high - CP_up_point[1])

        max_high = round(max_high)
    ''''''
    return up_tg1, up_tg2, up_tg3, up_tg5, max_high


def prepare_data(df, t1up, t4up):
    # Создаем scaler
    scaler = MinMaxScaler()

    # Выбираем интервал от t1up до t4up и вычисляем среднее между high и low
    interval_data = (df.loc[t1up[0]:t4up[0], 'high'] + df.loc[
                                                       t1up[0]:t4up[0],
                                                       'low']) / 2

    # Нормализуем данные
    y_data_normalized = scaler.fit_transform(
        np.array(interval_data).reshape(-1, 1))

    return y_data_normalized.ravel()

def compute_mse(df, normalized_data, start, end):
    # Вычисляем количество элементов в интервале
    num_elements = len(df.loc[start[0]:end[0]])

    # Выбираем соответствующие элементы из нормализованного массива
    normalized_interval_data = normalized_data[:num_elements]

    # Ваши данные
    x_data = np.array(range(len(normalized_interval_data)))

    # Подгоняем данные под параболу
    coeffs = np.polyfit(x_data, normalized_interval_data, deg=2)

    # Вычисляем предсказанные значения
    y_fit = np.polyval(coeffs, x_data)

    # Вычисляем среднеквадратичную ошибку
    mse = np.mean((normalized_interval_data - y_fit) ** 2)

    return mse

def calculate_risk_reward_ratio(entry_price, stop_price, take_price):
    """Считает соотношение риск/прибыль."""

    potential_risk = entry_price - stop_price
    potential_reward = take_price - entry_price

    if potential_risk != 0:  # Избегаем деления на ноль
        risk_reward_ratio = potential_reward / potential_risk
    else:
        risk_reward_ratio = float('inf')  # Бесконечность

    return risk_reward_ratio

def process_group_long(df, up_trend_points, ticker):
    """Обрабатывает группы точек т1-т3, находит торговые сетапы."""

    all_other_parameters_up = []

    for model in up_trend_points:
        t1up = model.t1up
        t2up = model.t2up
        t3up = model.t3up
        t4up = model.t4up
        t5up = model.t5up
        HP_up_point = model.HP_up_point
        LC_break_point = model.LC_break_point
        CP_up_point = model.CP_up_point
        dist_cp_t4_x1 = model.dist_cp_t4_x1

        # if not model.LT_break_point:
        #     continue
        #
        # if model.LT_break_point[0] > dist_cp_t4_x1:
        #     continue
        #
        # up_tg1, up_tg2, up_tg3, up_tg5, max_high = find_target_up(df, t1up, t4up, model.LT_break_point, model.CP_up_point)
        #
        # if up_tg1 < up_tg3 or up_tg1 < up_tg5:
        #     continue

        df_valid = df.loc[:t1up[0]][df.loc[:t1up[0]]['low'] < t1up[1]]

        first_bar_index_before_t1up = df_valid.last_valid_index()

        if first_bar_index_before_t1up is not None:
            first_bar_price = df.loc[
                first_bar_index_before_t1up, 'low']
            plt.plot(first_bar_index_before_t1up, first_bar_price,
                     color='pink',
                     marker='^',
                     markersize=10)
            num_of_distances_low_to_t1up = (t1up[
                                                0] - first_bar_index_before_t1up) / dist_cp_t4_x1

            plt.text(first_bar_index_before_t1up, first_bar_price,
                     num_of_distances_low_to_t1up,
                     fontsize=7,
                     color='b')
        else:
            num_of_distances_low_to_t1up = 1000

        df_valid = df.loc[:t4up[0]][df.loc[:t4up[0]]['high'] > t4up[1]]

        first_bar_index_before_t4up = df_valid.last_valid_index()

        if first_bar_index_before_t4up is not None:
            first_bar_price = df.loc[
                first_bar_index_before_t4up, 'high']
            plt.plot(first_bar_index_before_t4up, first_bar_price,
                     color='pink',
                     marker='>',
                     markersize=10)

            num_of_distances_high_to_t4up = (t4up[
                                                 0] - first_bar_index_before_t4up) / dist_cp_t4_x1
            plt.text(first_bar_index_before_t4up, first_bar_price,
                     num_of_distances_high_to_t4up,
                     fontsize=7,
                     color='b')
        else:
            num_of_distances_high_to_t4up = 1000

        # if num_of_distances_high_to_t4up < 0.025:
        #     continue
        #
        # if num_of_distances_low_to_t1up < 0.001:
        #     continue

        normalized_data = prepare_data(df, t1up, t4up)

        mse_t1_t2 = compute_mse(df, normalized_data, t1up, t2up)
        # print('t1up to t2up mse:', mse_t1_t2)

        mse_t2_t3 = compute_mse(df, normalized_data, t2up, t3up)
        # print('t2up to t3up mse:', mse_t2_t3)

        mse_t3_t4 = compute_mse(df, normalized_data, t3up, t4up)

        mse_t1_t3 = compute_mse(df, normalized_data, t1up, t3up)
        # print('t1up to t3up mse:', mse_t1_t3)

        mse = mse_t1_t2 + mse_t2_t3
        # print(mse)

        # Рассчитаем отношения
        ratio1 = (t2up[0] - t1up[0]) / (t3up[0] - t1up[0])
        ratio2 = (t4up[0] - t3up[0]) / (t4up[0] - t1up[0])

        # Золотое сечение
        golden_ratio = (1 + 5 ** 0.5) / 2  # равно примерно 1.618

        # Сравним наши отношения со золотым сечением
        print(abs(ratio1 - golden_ratio))
        print(abs(ratio2 - golden_ratio))

        ratio1_golden = ratio1 - golden_ratio
        ratio2_golden = ratio2 - golden_ratio
        ratio1_ratio2 = ratio1_golden / ratio2_golden
        ratio2_ratio1 = ratio2_golden / ratio1_golden

        CP_to_t1up = t1up[0] - CP_up_point[0]

        """"""

        lower_limit = int(t4up[0] + 1)
        upper_limit = int(
            model.dist_cp_t4_x2)  # or whatever upper limit you have defined

        # Вычисляем угловой коэффициент прямой между t1_up и t3_up
        slope = (t3up[1] - t1up[1]) / (t3up[0] - t1up[0])

        # Вычисляем значения прямой для всех точек между t1_up и t4_up
        line_values = t1up[1] + slope * (
                    np.arange(lower_limit, upper_limit) - t1up[0])

        # Находим индексы всех баров в нашем диапазоне, которые закрылись ниже прямой t1_up:t3_up
        bars_below_line = np.where(
            df['close'][lower_limit:upper_limit].values < line_values)

        # Если таких баров нет, bars_below_line будет пустым и мы пропускаем этот шаг цикла
        if bars_below_line[0].size == 0:
            continue

        # В противном случае, берем индекс первого такого бара, учитывая смещение на lower_limit
        first_bar_below_line = bars_below_line[0][0] + lower_limit

        """"""
        lower_limit = int(first_bar_below_line + 1)
        upper_limit = int(
            model.dist_cp_t4_x2)  # or whatever upper limit you have defined

        # Вычисляем угловой коэффициент прямой между t1_up и t3_up
        slope = (t3up[1] - t1up[1]) / (t3up[0] - t1up[0])

        # Вычисляем значения прямой для всех точек между t1_up и t4_up
        line_values = t1up[1] + slope * (
                np.arange(lower_limit, upper_limit) - t1up[0])

        # Расстояние между t1_up[1] и t4_up[1]
        dist_t1_t4 = abs(t4up[1] - t1up[1])

        # Увеличиваем значения линии на 5% от dist_t1_t4
        line_values += 0.05 * dist_t1_t4

        # Находим индексы всех баров в нашем диапазоне, которые закрылись ниже прямой t1_up:t3_up
        bars_below_line = np.where(
            df['close'][lower_limit:upper_limit].values > line_values)

        # Если таких баров нет, или их меньше двух, bars_below_line будет пустым и мы пропускаем этот шаг цикла
        if bars_below_line[0].size < 2:
            continue

        # В противном случае, берем индексы первых двух таких баров, учитывая смещение на lower_limit
        second_bars_below_line = bars_below_line[0][1] + lower_limit

        """"""

        lower_limit = int(second_bars_below_line + 1)
        upper_limit = int(
            model.dist_cp_t4_x2)  # or whatever upper limit you have defined

        # Вычисляем угловой коэффициент прямой между t1_up и t3_up
        slope = (t3up[1] - t1up[1]) / (t3up[0] - t1up[0])

        # Вычисляем значения прямой для всех точек между t1_up и t4_up
        line_values = t1up[1] + slope * (
                    np.arange(lower_limit, upper_limit) - t1up[0])

        # Находим индексы всех баров в нашем диапазоне, которые закрылись ниже прямой t1_up:t3_up
        bars_below_line = np.where(
            df['low'][lower_limit:upper_limit].values <= line_values)

        # Если таких баров нет, bars_below_line будет пустым и мы пропускаем этот шаг цикла
        if bars_below_line[0].size == 0:
            continue

        # В противном случае, берем индекс первого такого бара, учитывая смещение на lower_limit
        first_bar_below_line = bars_below_line[0][0] + lower_limit

        entry_index = first_bar_below_line

        intersection_price = t1up[1] + slope * (
                    first_bar_below_line - t1up[0])

        entry_price = intersection_price

        min_low = df['low'][int(t4up[0]):int(entry_index)].min()

        stop_price = min_low
        # up_take_lines = analysis_base.up_take_lines

        up_take_200 = t4up[1] + (t4up[1] - t1up[1]) * 2

        take_price = t4up[1] + 0.9 * (
                up_take_200 - t4up[1])

        take_price = model.up_take_lines[1]
        percent_difference = abs(
            ((entry_price - take_price) / entry_price) * 100
        )

        risk_reward_ratio = calculate_risk_reward_ratio(entry_price,
                                                        stop_price,
                                                        take_price)

        # if percent_difference < 0.4 or risk_reward_ratio < 1 or risk_reward_ratio > 5:
        #     continue

        '''Получаем дополнительные свойства модели.'''
        analysis_base = BaseTradeAnalysis(t1up, t2up, t3up)
        # Создаем объект для расчетов дополнительных
        # свойств модели
        up_enter_point1 = (t3up[0] + 1, t2up[1] * 1.2)
        model_property = UpModelProperty(df, up_enter_point1,
                                         analysis_base)

        # Геометрические расчеты
        angle_deg_LT, _, _, _ = model_property.get_LT_up
        max_distance, _ = model_property.find_point_t1_t2

        '''На основе предыдущих шагов производим расчеты,
        для получения дополнительных параметров.'''
        # Создаем объект расчета дополнительных параметров
        analysis_advanced = AdvancedTradeAnalysis(
            model_property)

        # Получаем различные статистические данные,
        # геометрические измерения
        (percentage_difference_min_low_and_t1,
         percentage_difference_min_close_and_t1,
         percentage_difference_max_close_and_t1,
         diff_price_change, length3_point_t1_t2_norm,
         angle_t2_t3, area_under_curve
         ) = analysis_advanced.calculate_property

        (angle_t3_enter,
         radius_curvature_t2_t3_enter
         ) = analysis_advanced.calculate_param_t2_t3_up_enter

        if radius_curvature_t2_t3_enter > 0.7:
            continue
        # Вычисляем соотношение хвоста и тела свечи
        # для различных точек
        (tail_to_body_ratio_t1,
         tail_to_body_ratio_t2,
         tail_to_body_ratio_t3,
         tail_to_body_ratio_enter_point_back_1
         ) = analysis_advanced.candle_tail_body_parameters

        # Рассчитываем std_dev_y_mean_y
        (std_dev_y_mean_y,
         std_dev_y_mean_y_1,
         std_dev_y_mean_y_2,
         std_dev_y_mean_y_3,
         std_dev_y_mean_y_4,
         std_dev_y_mean_y_5,
         std_dev_y_mean_y_6,
         std_dev_y_mean_y_7,
         std_dev_y_t2_t3_up_enter
         ) = analysis_advanced.compute_std_dev_y_mean_y
        """
        # if angle_t2_t3 < -0.0169:
        #     continue
        # if std_dev_y_mean_y_6 < 0.0046:
        #     continue
        """

        # if std_dev_y_mean_y_5 < 0.094:
        #     continue
        # Получаем значения индикаторов,
        # делаем расчеты на их основе
        # RSI
        (rsi_value1,
         rsi_value2,
         rsi_value3,
         rsi_value_enter
         ) = analysis_advanced.get_rsi

        # # VWAP
        (vwap_t1,
         vwap_enter,
         vwap_ratio_t1,
         vwap_ratio_enter
         ) = analysis_advanced.get_vwap

        # Симулируем торговлю
        stop_point_found = False
        take_point_found = False
        up_stop_point = None
        up_take_point = None

        entry_date = df.loc[entry_index, 'dateTime']

        # Формируем переменную точки входа
        up_enter_point = (entry_index, entry_price)
        print(f'Точка входа найдена. {entry_date}')

        # plt.plot(t4up[0], t4up[1], color='red',
        #          marker='^',
        #          markersize=10, markeredgecolor='black')

        plt.plot(up_enter_point[0], up_enter_point[1], color='red',
                 marker='^',
                 markersize=10, markeredgecolor='black')
        plt.text(up_enter_point[0], up_enter_point[1],
                 'up_enter_point {:.2f}'.format(up_enter_point[1]),
                 fontsize=7,
                 color='b')
        plt.text(up_enter_point[0], up_enter_point[1], entry_date,
                 rotation='vertical')

        # Для наглядности маркируем точку входа на графике
        # plt.plot(up_enter_point[0], up_enter_point[1],
        #          color='red', marker='^',
        #          markersize=10, markeredgecolor='black')
        # plt.text(up_enter_point[0], up_enter_point[1],
        #          'up_enter_point {:.2f}'.format(up_enter_point[1]),
        #          fontsize=7, color='b')
        # plt.text(up_enter_point[0], up_enter_point[1], entry_date,
        #          color='black', rotation='vertical', va='top')

        close_index = entry_index + 100
        upper_limit = min(close_index, len(df))
        for g in range(up_enter_point[0] + 1, upper_limit):

            if df.loc[g, 'low'] <= stop_price:
                stop_point_found = True
                stop_point_g = g
                break

        for g in range(up_enter_point[0] + 1, upper_limit):
            if df.loc[g, 'high'] >= take_price:
                take_point_found = True
                take_point_g = g
                break

        if take_point_found and (not stop_point_found
                                 or take_point_g < stop_point_g):
            up_take_point = (take_point_g, take_price)
            print('Сделка закрыта по тейку')

            plt.plot(up_take_point[0], up_take_point[1],
                     color='green', marker='^',
                     markersize=10,
                     markeredgecolor='black')
            plt.text(up_take_point[0], up_take_point[1],
                     'up_take_point',
                     fontsize=7, color='b')

        elif stop_point_found and (not take_point_found
                                   or stop_point_g < take_point_g):
            up_stop_point = (stop_point_g, stop_price)

            stop_point_found = True
            print('Сделка закрыта по стопу')
            plt.plot(up_stop_point[0], up_stop_point[1],
                     color='yellow',
                     marker='^',
                     markersize=10,
                     markeredgecolor='black')
            plt.text(up_stop_point[0], up_stop_point[1],
                     'up_stop_point {:.2f}'.
                     format(up_stop_point[1]),
                     fontsize=7, color='b')

        else:
            if close_index < len(df):
                up_close_stop_point = (
                    close_index, df.loc[close_index, 'close'])
            else:
                up_close_stop_point = None
            # else:
            #     up_close_stop_point = (
            #     len(df) - 1, df.loc[len(df) - 1, 'close'])
            # plt.plot(up_stop_point[0], up_stop_point[1],
            #          color='yellow',
            #          marker='^',
            #          markersize=15,
            #          markeredgecolor='black')

            print("Ни одна из цен не была достигнута.")

        # Рисуем линию тейка
        # Инициализация up_take_lines
        # up_take_lines = up_take_lines
        # plt.axvline(x=up_take_lines[0], color='g', linestyle='--',
        #             linewidth=0.5)
        # # Количество баров т1-т3
        # dist_t1_t3_bar = t3up[0] - t1up[0]
        # # Конечная точка линии
        # end_point = up_take_lines[0] + dist_t1_t3_bar * 3
        # plt.plot([up_take_lines[0], end_point], [up_take_lines[1],
        #                                          up_take_lines[1]],
        #          color='g',
        #          linestyle='--', linewidth=0.5)
        # # Добавляем текст
        # plt.text(up_take_lines[0], up_take_lines[1], 'TAKE LINE',
        #          color='g', fontsize=8, va='bottom')

        # Инициализация переменной
        close_position_point = None
        close_position_point_price = None
        diff = None
        profit_or_lose = None
        # Проверка условий
        if up_stop_point is not None:
            close_position_point = up_stop_point
        elif up_take_point is not None:
            close_position_point = up_take_point
        else:
            if up_close_stop_point:
                close_position_point = up_close_stop_point
            else:
                close_position_point = None

        if close_position_point is not None:

            diff = (
                    (close_position_point[1] -
                     up_enter_point[1])
                    / up_enter_point[1] * 100)

            profit_or_lose = None
            if diff is not None:
                if diff > 0:
                    profit_or_lose = 1
                else:
                    profit_or_lose = 0
            if close_position_point is not None:
                close_position_point_price = close_position_point[
                    1]
            else:
                close_position_point_price = None

        open_to_close_trade_duration = None
        if close_position_point_price is not None:
            open_to_close_trade_duration = close_position_point[0] - \
                                           up_enter_point[0]
        model_interval = t4up[0] - t1up[0]

        other_parameters = (
            # Информация о сделке
            entry_date,
            entry_price,
            stop_price,
            take_price,
            close_position_point_price,
            diff,
            profit_or_lose,
            risk_reward_ratio,
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
            vwap_ratio_enter,

            open_to_close_trade_duration,
            model_interval,

            ticker,

            ratio1_golden,
            ratio2_golden,
            ratio1_ratio2,
            ratio2_ratio1,

            CP_to_t1up,

            num_of_distances_low_to_t1up,
            num_of_distances_high_to_t4up,
            mse_t1_t2,
            mse_t2_t3,
            mse_t3_t4,
            mse_t1_t3,
            mse
        )

        all_other_parameters_up.append(other_parameters)

    return all_other_parameters_up