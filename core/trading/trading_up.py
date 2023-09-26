import logging

import numpy as np
import matplotlib.pyplot as plt

from core.base_trade_analysis_long import BaseTradeAnalysis
from core.up_model_property import UpModelProperty
from core.advanced_trade_analysis_long import AdvancedTradeAnalysis
from core.upmodelclass import UpModel
from core.up_exp_property import Line, Point


def calculate_risk_reward_ratio(entry_price, stop_price, take_price):
    """Считает соотношение риск/прибыль."""

    potential_risk = entry_price - stop_price
    potential_reward = take_price - entry_price

    if potential_risk != 0:  # Избегаем деления на ноль
       risk_reward_ratio = potential_reward / potential_risk
    else:
       risk_reward_ratio = float('inf')  # Бесконечность

    return risk_reward_ratio

def process_group_long(df, up_trend_points):
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

        if not HP_up_point:
            continue

        # if model.LC_break_point:
        #     if model.LC_break_point[0] <= HP_up_point[0]:
        #         continue

        """Проверка достижения HP"""

        complete_HP = t4up[1] + 0.9 * (
                HP_up_point[1] - t4up[1])
        filtered_df = df.loc[t4up[0] + 1: len(df)]
        high_than_HP = filtered_df[filtered_df['high'] >= complete_HP]

        # проверим, что lower_than_t2up не пуст
        if not high_than_HP.empty:
            HP_complete_index = high_than_HP.index[0]
            plt.plot(HP_complete_index, HP_up_point[1], 'yo', markersize=5)
        else:
            continue

        if LC_break_point:
            if LC_break_point[0] < HP_complete_index:
                continue

        # """Поиск лоя ниже НР"""
        # filtered_df = df.loc[HP_complete_index+1: len(df)]
        # low_than_HP = filtered_df[filtered_df['low'] < HP_up_point[1]]
        #
        # # проверим, что lower_than_t2up не пуст
        # # if not low_than_HP.empty:
        # #     HP_low_after_complete = low_than_HP.index[0]
        # #     plt.plot(HP_low_after_complete, HP_up_point[1], 'ro', markersize=5)
        # # else:
        # #     continue
        #
        # if len(low_than_HP) >= 1:
        #     HP_low_after_complete = low_than_HP.index[0]
        #     plt.plot(HP_low_after_complete, HP_up_point[1], 'ro', markersize=5)
        # else:
        #     continue
        #
        # # """Проверка достижения HP"""
        # #
        # # filtered_df = df.loc[HP_low_after_complete+1: model.dist_cp_t4_x2]
        # # high_than_HP = filtered_df[filtered_df['high'] >= HP_up_point[1]]
        # #
        # # # проверим, что lower_than_t2up не пуст
        # # if not high_than_HP.empty:
        # #     high_than_HP = high_than_HP.index[0]
        # #     plt.plot(high_than_HP, HP_up_point[1], 'yo', markersize=5)
        # # else:
        # #     continue
        # #
        # # max_high_after_complete_HP = max(
        # #     df.loc[t4up[0]:high_than_HP, 'high'])
        # # # HP_up_point_100 = HP_up_point[1] + (HP_up_point[1] - t1up[1]) * 1
        #
        # max_high_after_complete_HP = max(
        #     df.loc[t4up[0]:HP_low_after_complete, 'high'])

        entry_index = HP_complete_index

        entry_price = HP_up_point[1]

        stop_HP = t4up[1] + 1.9 * (
                HP_up_point[1] - t4up[1])
        stop_price = stop_HP
        # up_take_lines = analysis_base.up_take_lines

        take_price = t4up[1] + 0.1 * (
                HP_up_point[1] - t4up[1])
        # take_price = t4up[1] + 0.9 * (
        #         up_enter_point_pre - t4up[1])

        percent_difference = abs(
            ((entry_price - take_price) / entry_price) * 100
        )

        risk_reward_ratio = calculate_risk_reward_ratio(entry_price,
                                                        stop_price, take_price)

        if percent_difference < 0.4 or risk_reward_ratio < 0.5 or risk_reward_ratio > 5:
            continue









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

        # Получаем значения индикаторов,
        # делаем расчеты на их основе
        # RSI
        # (rsi_value1,
        #  rsi_value2,
        #  rsi_value3,
        #  rsi_value_enter
        #  ) = analysis_advanced.get_rsi
        #
        # # VWAP
        # (vwap_t1,
        #  vwap_enter,
        #  vwap_ratio_t1,
        #  vwap_ratio_enter
        #  ) = analysis_advanced.get_vwap

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

        for g in range(up_enter_point[0], len(df)):

            if df.loc[g, 'high'] >= stop_price:
                stop_point_found = True
                stop_point_g = g
                break

        for g in range(up_enter_point[0]+1, len(df)):
            if df.loc[g, 'low'] <= take_price:
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
            close_position_point = None

        if close_position_point is not None:

            diff = (
                           (close_position_point[1] -
                            up_enter_point[1])
                           / up_enter_point[1] * 100) * -1

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
            # rsi_value1,
            # rsi_value2,
            # rsi_value3,
            # rsi_value_enter,
            # vwap_ratio_t1,
            # vwap_ratio_enter,

            # open_to_close_trade_duration,
            # model_interval

        )

        all_other_parameters_up.append(other_parameters)

    return all_other_parameters_up
