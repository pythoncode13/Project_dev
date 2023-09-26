import logging

import numpy as np
import matplotlib.pyplot as plt

from core.models_trade_property.base_trade_analysis_short import BaseTradeAnalysis
from core.models_trade_property.down_model_property import DownModelProperty
from core.models_trade_property.advanced_trade_analysis_short import AdvancedTradeAnalysis
from core.models.upmodelclass import UpModel
from core.models.up_model_property import Line, Point
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(df, t1down, t4down):
    # Создаем scaler
    scaler = MinMaxScaler()

    # Выбираем интервал от t1down до t4down и вычисляем среднее между high и low
    interval_data = (df.loc[t1down[0]:t4down[0], 'high'] + df.loc[t1down[0]:t4down[0], 'low']) / 2

    # Нормализуем данные
    y_data_normalized = scaler.fit_transform(np.array(interval_data).reshape(-1, 1))

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



def calculate_risk_reward_ratio_short(entry_price, take_price, stop_price):

    potential_risk = abs(stop_price - entry_price)
    potential_reward = abs(entry_price - take_price)

    if potential_risk != 0:  # Проверка на ноль, чтобы избежать ошибки деления на ноль
        risk_reward_ratio = potential_reward / potential_risk
    else:
        risk_reward_ratio = float('inf')  # Бесконечность
    return risk_reward_ratio


def process_group_short(df, up_trend_points, ticker):
    """Обрабатывает группы точек т1-т3, находит торговые сетапы."""

    all_other_parameters_down = []

    for model in up_trend_points:
        t1down = model.t1down
        t2down = model.t2down
        t3down = model.t3down
        t4down = model.t4down
        t5down = model.t5down
        HP_down_point = model.HP_down_point
        LC_break_point = model.LC_break_point
        CP_down_point = model.CP_down_point
        dist_cp_t4_x1 = model.dist_cp_t4_x1

        # if (t3down[0] - t1down[0]) < (t1down[0] - CP_down_point[0]):
        #     continue

        df_valid = df.loc[:t1down[0]][df.loc[:t1down[0]]['high'] > t1down[1]]

        first_bar_index_before_t1down = df_valid.last_valid_index()

        if first_bar_index_before_t1down is not None:
            first_bar_price = df.loc[first_bar_index_before_t1down, 'high']
            plt.plot(first_bar_index_before_t1down, first_bar_price, color='pink',
                     marker='^',
                     markersize=10)
            num_of_distances_low_to_t1down = (t1down[0] - first_bar_index_before_t1down) / dist_cp_t4_x1

            plt.text(first_bar_index_before_t1down, first_bar_price, num_of_distances_low_to_t1down,
                     fontsize=7,
                     color='b')
        else:
            num_of_distances_low_to_t1down = 1000



        df_valid = df.loc[:t4down[0]][df.loc[:t4down[0]]['low'] < t4down[1]]

        first_bar_index_before_t4down = df_valid.last_valid_index()

        if first_bar_index_before_t4down is not None:
            first_bar_price = df.loc[first_bar_index_before_t4down, 'low']
            plt.plot(first_bar_index_before_t4down, first_bar_price, color='pink',
                     marker='>',
                     markersize=10)

            num_of_distances_high_to_t4down = (t4down[0] - first_bar_index_before_t4down) / dist_cp_t4_x1
            plt.text(first_bar_index_before_t4down, first_bar_price,
                     num_of_distances_high_to_t4down,
                     fontsize=7,
                     color='b')
        else:
            num_of_distances_high_to_t4down = 1000

        # if num_of_distances_high_to_t4down < 0.025:
        #     continue
        #
        # if num_of_distances_low_to_t1down < 0.001:
        #     continue

        print('t1down[0], t2down[0], t3down[0], t4down[0]', t1down[0], t2down[0], t3down[0], t4down[0])
        normalized_data = prepare_data(df, t1down, t4down)
        print('normalized_data:', normalized_data)

        mse_t1_t2 = compute_mse(df, normalized_data, t1down, t2down)
        print('t1down to t2down mse:', mse_t1_t2)

        mse_t2_t3 = compute_mse(df, normalized_data, t2down, t3down)
        print('t2down to t3down mse:', mse_t2_t3)

        mse_t3_t4 = compute_mse(df, normalized_data, t3down, t4down)

        mse_t1_t3 = compute_mse(df, normalized_data, t1down, t3down)
        print('t1down to t3down mse:', mse_t1_t3)

        mse = mse_t1_t2 + mse_t2_t3
        # print(mse)

        if mse_t3_t4 < 0.0001:
            continue

        # Рассчитаем отношения
        ratio1 = (t2down[0] - t1down[0]) / (t3down[0] - t1down[0])
        ratio2 = (t4down[0] - t3down[0]) / (t4down[0] - t1down[0])

        # Золотое сечение
        golden_ratio = (1 + 5 ** 0.5) / 2  # равно примерно 1.618

        # Сравним наши отношения со золотым сечением
        print(abs(ratio1 - golden_ratio))
        print(abs(ratio2 - golden_ratio))

        ratio1_golden = ratio1 - golden_ratio
        ratio2_golden = ratio2 - golden_ratio
        ratio1_ratio2 = ratio1_golden / ratio2_golden
        ratio2_ratio1 = ratio2_golden / ratio1_golden

        CP_to_t1down = t1down[0] - CP_down_point[0]
        # if HP_down_point:
        #     continue

        # if LC_break_point:
        #     if LC_break_point[0] >= HP_down_point[0]:
        #         continue

        # upper_limit = min(int(model.dist_cp_t4_x2), len(df))
        #
        # # Находим нижний край тела каждой свечи
        # lower_body_edge = df[['open', 'close']].min(axis=1)
        #
        # # Находим индекс первого бара, нижний край которого закрылся выше уровня model.up_take_lines[1]
        # first_bar_above_t4down = np.argmax(
        #     lower_body_edge > model.up_take_lines[1])
        #
        # # Если такого бара не существует, выходим из цикла
        # if first_bar_above_t4down is None:
        #     continue

        # # В противном случае обновляем t4down
        # t4down = first_bar_above_t4down

        # Находим нижний край тела каждой свечи
        lower_body_edge = df[['low']].min(axis=1)

        # # Задаем диапазон поиска
        # lower_limit = int(t4down[0]+1)
        # # upper_limit = min(int(model.dist_cp_t4_x1), len(df))
        # upper_limit = len(df)
        #
        # # Находим индексы всех баров в нашем диапазоне, нижний край которых закрылся выше уровня model.up_take_lines[1]
        # bars_above_t4down = np.where(
        #     lower_body_edge[lower_limit:upper_limit] < t4down[1])
        #
        # plt.plot(t4down[0], t4down[1], color='red',
        #          marker='^',
        #          markersize=10, markeredgecolor='black')
        #
        # # Если таких баров нет, bars_above_t4down будет пустым и мы пропускаем этот шаг цикла
        # if bars_above_t4down[0].size == 0:
        #     continue
        # print("тест")
        # # В противном случае, берем индекс первого такого бара, учитывая смещение на lower_limit
        # first_bar_above_t4down = bars_above_t4down[0][0] + lower_limit

        # Задаем диапазон поиска
        lower_limit = int(t4down[0] + 1)
        upper_limit = min(int(model.dist_cp_t4_x1), len(df))

        # Создаем булевой массив, где True означает, что бар находится ниже t4down[1]
        is_below_t4down = lower_body_edge[lower_limit:upper_limit] < t4down[1]

        # Используем np.argmax() для нахождения первого бара ниже t4down[1], если таковой существует
        first_bar_below_t4down = np.argmax(is_below_t4down) if np.any(
            is_below_t4down) else None

        if first_bar_below_t4down is not None:
            # Добавляем смещение lower_limit к индексу
            entry_index = first_bar_below_t4down + lower_limit
        else:
            continue  # Если таких баров нет, пропускаем шаг цикла

        # entry_index = first_bar_below_t4down + lower_limit

        entry_price = t4down[1]
        # 4stop_price = t1down[1]

        stop_price = t1down[1]
        # up_take_lines = analysis_base.up_take_lines

        up_take_200 = t4down[1] - (t1down[1] - t4down[1]) * 2

        take_price = t4down[1] + 0.9 * (
                up_take_200 - t4down[1])

        stop_percent_difference = stop_price * 100 / entry_price - 100
        take_percent_difference = 100 - take_price * 100 / entry_price

        percent_difference = ((entry_price - take_price) / entry_price) * 100  # разница для шорт позиций

        risk_reward_ratio = calculate_risk_reward_ratio_short(entry_price, take_price,
                                                    stop_price)

        # if percent_difference < 0.4 or risk_reward_ratio < 1 or risk_reward_ratio > 5:
        #     continue

        '''Получаем дополнительные свойства модели.'''
        analysis_base = BaseTradeAnalysis(t1down, t2down, t3down)
        # Создаем объект для расчетов дополнительных
        # свойств модели

        down_enter_point1 = (int(t3down[0]) + 1, t2down[1] * 1.2)
        model_property = DownModelProperty(df, down_enter_point1,
                                         analysis_base)

        # Геометрические расчеты
        angle_deg_LT, _, _, _ = model_property.get_LT_down
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
         ) = analysis_advanced.calculate_param_t2_t3_down_enter

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
         vwap_ratio_enter,
         vwap_t1_v2,
         vwap_enter_v2,
         vwap_ratio_t1_v2,
         vwap_ratio_enter_v2,
         ) = analysis_advanced.get_vwap

        # Симулируем торговлю
        stop_point_found = False
        take_point_found = False
        down_stop_point = None
        down_take_point = None

        entry_date = df.loc[entry_index, 'dateTime']

        # Формируем переменную точки входа
        down_enter_point = (entry_index, entry_price)
        print(f'Точка входа найдена. {entry_date}')

        # plt.plot(t4down[0], t4down[1], color='red',
        #          marker='^',
        #          markersize=10, markeredgecolor='black')

        plt.plot(down_enter_point[0], down_enter_point[1], color='red',
                 marker='^',
                 markersize=10, markeredgecolor='black')
        plt.text(down_enter_point[0], down_enter_point[1],
                 'up_enter_point {:.2f}'.format(down_enter_point[1]),
                 fontsize=7,
                 color='b')
        plt.text(down_enter_point[0], down_enter_point[1], entry_date,
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
        for g in range(down_enter_point[0], upper_limit):

            if df.loc[g, 'high'] >= stop_price:
                stop_point_found = True
                stop_point_g = g
                break

        for g in range(down_enter_point[0] + 1, upper_limit):
            if df.loc[g, 'low'] <= take_price:
                take_point_found = True
                take_point_g = g
                break

        if take_point_found and (not stop_point_found
                                 or take_point_g < stop_point_g):
            down_take_point = (take_point_g, take_price)
            print('Сделка закрыта по тейку')

            plt.plot(down_take_point[0], down_take_point[1],
                     color='green', marker='^',
                     markersize=10,
                     markeredgecolor='black')
            plt.text(down_take_point[0], down_take_point[1],
                     'up_take_point',
                     fontsize=7, color='b')

        elif stop_point_found and (not take_point_found
                                   or stop_point_g < take_point_g):
            down_stop_point = (stop_point_g, stop_price)

            stop_point_found = True
            print('Сделка закрыта по стопу')
            plt.plot(down_stop_point[0], down_stop_point[1],
                     color='yellow',
                     marker='^',
                     markersize=10,
                     markeredgecolor='black')
            plt.text(down_stop_point[0], down_stop_point[1],
                     'up_stop_point {:.2f}'.
                     format(down_stop_point[1]),
                     fontsize=7, color='b')

        else:
            if close_index < len(df):
                down_close_stop_point = (
                close_index, df.loc[close_index, 'close'])
            else:
                down_close_stop_point = None
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
        # dist_t1_t3_bar = t3down[0] - t1down[0]
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
        if down_stop_point is not None:
            close_position_point = down_stop_point
        elif down_take_point is not None:
            close_position_point = down_take_point
        else:
            if down_close_stop_point:
                close_position_point = down_close_stop_point
            else:
                close_position_point = None

        if close_position_point is not None:

            diff = ((down_enter_point[1] - close_position_point[1]) /
                    down_enter_point[1] * 100)

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
                                           down_enter_point[0]
        model_interval = t4down[0] - t1down[0]

        other_parameters = (
            # Информация о сделке
            entry_date,
            ticker,
            open_to_close_trade_duration,
            entry_price,
            stop_price,
            take_price,
            close_position_point_price,
            diff,
            profit_or_lose,
            stop_percent_difference,
            take_percent_difference,
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
            vwap_t1_v2,
            vwap_enter_v2,
            vwap_ratio_t1_v2,
            vwap_ratio_enter_v2,
            vwap_ratio_t1,
            vwap_ratio_enter,

            model_interval,

            ratio1_golden,
            ratio2_golden,
            ratio1_ratio2,
            ratio2_ratio1,

            CP_to_t1down,

            num_of_distances_low_to_t1down,
            num_of_distances_high_to_t4down,
            mse_t1_t2,
            mse_t2_t3,
            mse_t3_t4,
            mse_t1_t3,
            mse
        )

        all_other_parameters_down.append(other_parameters)

    return all_other_parameters_down