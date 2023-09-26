import logging

import numpy as np
import matplotlib.pyplot as plt

from core.base_trade_analysis_long import BaseTradeAnalysis
from core.up_model_property import UpModelProperty
from core.advanced_trade_analysis_long import AdvancedTradeAnalysis


def process_group_long(df, up_trend_points):
    """Обрабатывает группы точек т1-т3, находит торговые сетапы."""

    all_other_parameters_up = []

    for model in up_trend_points:
        t1up = model.t1up
        t2up = model.t2up
        t3up = model.t3up
        t4up = model.t4up
        # if model.LT_up_breakout_point is not None and model.dist_cp_t4_x2 is not None and \
        #         model.LT_up_breakout_point[0] < model.dist_cp_t4_x2:
        #     continue
        if model.above_is_faster_breakout() is False:
            continue
        result = model.find_t5up()
        if result is None or result[0] is None:
            continue
        t5up = result[0]
        plt.plot(t5up[0], t5up[1], color='red',
                 marker='>',
                 markersize=10, markeredgecolor='black')
        t2up_candle = df.loc[t2up[0]]
        t5up_candle = df.loc[t5up[0]]

        t2up_upper_body_edge = max(t2up_candle['open'], t2up_candle['close'])
        t5up_lower_body_edge = min(t5up_candle['open'], t5up_candle['close'])

        if t2up_upper_body_edge > t5up_lower_body_edge:
            continue
        HP_up_point = model.find_t5up()[1]
        if HP_up_point is None:
            continue
        HP_up_point_100 = HP_up_point[1] + (HP_up_point[1] - t1up[1]) * 1
        # Получает все данные закрытия между t4up и некоторой другой точкой
        close_between_LC = df.loc[
                           t4up[0]+1: t4up[0] + (t4up[0] - model.CP_up_point[0]),
                           'close']

        # Создает булев массив, где True означает, что цена закрытия выше прямой
        intersects_between_index = close_between_LC > (
                    model.slope_LC * close_between_LC.index + model.intersect_LC)

        # Если нет ни одного True значения в intersects_between_index, возвращаемся к началу цикла
        if not intersects_between_index.any():
            continue

        # Находит первую строку, где close был выше линии
        up_enter_point_pre = close_between_LC[intersects_between_index].head(1)

        # Извлекает индекс (дату) и цену закрытия этого бара
        entry_index = up_enter_point_pre.index[0]
        entry_price = up_enter_point_pre.values[0]

        # Создает кортеж из индекса и цены
        up_enter_point = (entry_index, entry_price)

        analysis_base = BaseTradeAnalysis(t1up, t2up, t3up)

        # entry_adjustment = 0.20  # Изменение цены входа на 10%
        # price_difference = abs(t4up[1] - t5up[1])  # Отрезок "стоп - вход"
        # # t4up[1] + entry_adjustment * price_difference
        #
        entry_price = up_enter_point[1]
        # # stop_price = analysis_base.stop_price
        #
        # stop_adjustment = 0.15  # Изменение стопа на 5%
        # price_difference = abs(t4up[1] - t5up[1])  # Отрезок "стоп - вход"
        stop_price = t5up[1]
        # up_take_lines = analysis_base.up_take_lines
        take_price = HP_up_point_100
        up_take_100 = t4up[1] + (t4up[1] - t1up[1]) * 1
        up_take_200 = t4up[1] + (t4up[1] - t1up[1]) * 2
        if HP_up_point[1] >= up_take_100:
            continue
        plt.hlines(y=HP_up_point[1], xmin=HP_up_point[0], xmax=t4up[0] + 10,
                   colors='purple', linestyles='solid',
                   linewidth=0.5)
        plt.hlines(y=HP_up_point_100, xmin=HP_up_point[0], xmax=t4up[0] + 10,
                                  colors='purple', linestyles='solid',
                                  linewidth=0.5)
        print('HP_up_point', HP_up_point)
        print('up_take_200', up_take_200)
        # if HP_up_point is not None:
        #     plt.hlines(y=HP_up_point[1], xmin=t4up[0], xmax=t4up[0] + 10,
        #                colors='purple', linestyles='solid',
        #                linewidth=0.5)
        #     LC_HP = result[2]
        #     plt.plot(LC_HP[0], LC_HP[1], ':', color='purple',
        #              linewidth=0.9)
        #     if up_take_100 < HP_up_point[1]:
        #         take_price = up_take_100
        #     else:
        #         take_price = HP_up_point[1]
        # else:
        #     take_price = up_take_100

        percent_difference = abs(
            ((entry_price - take_price) / entry_price) * 100
        )
        potential_risk = entry_price - stop_price
        potential_reward = take_price - entry_price
        risk_reward_ratio = potential_reward / potential_risk

        # percent_difference = analysis_base.percent_difference
        # risk_reward_ratio = analysis_base.risk_reward_ratio

        if percent_difference < 0.4 or risk_reward_ratio < 0.8:
            continue

        '''Получаем дополнительные свойства модели.'''
        # Создаем объект для расчетов дополнительных
        # свойств модели
        up_enter_point = (t3up[0]+1, t2up[1]*1.2)
        model_property = UpModelProperty(df, up_enter_point,
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
        (rsi_value1,
         rsi_value2,
         rsi_value3,
         rsi_value_enter
         ) = analysis_advanced.get_rsi

        # VWAP
        (vwap_t1,
         vwap_enter,
         vwap_ratio_t1,
         vwap_ratio_enter
         ) = analysis_advanced.get_vwap

        # Симулируем торговлю
        stop_point_found = False
        take_point_found = False
        take_point_g = None
        stop_point_g = None


        # Извлекаем нужные данные заранее
        t1up_0_plus_1 = t1up[0] + 1
        t3up_0 = t3up[0]
        t2up_0_minus_1 = t2up[0] - 1
        t1up_1 = t1up[1]
        t2up_1 = t2up[1]

        # # Проверяем условия
        # conditions = [
        #     t1up_0_plus_1 == t2up[0],
        #     (df.loc[t1up_0_plus_1:t3up_0, 'low'] <= t1up_1).any(),
        #     (df.loc[t2up[0] + 1:t3up_0, 'high'] >= t2up_1).any(),
        #     (df.loc[t1up_0_plus_1:t2up_0_minus_1, 'high'] >= t2up_1).any()
        # ]
        #
        # # Проверяем условия соответствия модели
        # if any(conditions):
        #     continue

        up_stop_point = None
        up_take_point = None
        return_in_model = None

        # for j in np.arange(model.LT_up_breakout_point[0], model.dist_cp_t4_x2):
        #
        #     # Проверка условия для входа в сделку
        #     y_value_at_bar = model.slope_LT * j + model.intercept_LT
        #     if df.loc[j, 'high'] > y_value_at_bar:
        #         return_in_model = j
        #         break
        #
        # if return_in_model is not None:
        #     for j in np.arange(return_in_model, len(df)):
        up_enter_point = (entry_index, entry_price)
        entry_index = up_enter_point[0]
        entry_date = df.loc[entry_index, 'dateTime']

        # Формируем переменную точки входа
        up_enter_point = (entry_index, entry_price)
        logging.debug(f'Точка входа найдена. {entry_date}')

        plt.plot(t4up[0], t4up[1], color='red',
                 marker='^',
                 markersize=10, markeredgecolor='black')

        plt.plot(up_enter_point[0], up_enter_point[1], color='red',
                 marker='^',
                 markersize=3, markeredgecolor='black')
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

        for g in range(up_enter_point[0] + 1, len(df)):

            if df.loc[g, 'low'] <= stop_price:
                stop_point_found = True
                stop_point_g = g
                break

        for g in range(up_enter_point[0], len(df)):
            if df.loc[g, 'high'] >= take_price:
                take_point_found = True
                take_point_g = g
                break

        if take_point_found and (not stop_point_found
                                 or take_point_g < stop_point_g):
            up_take_point = (take_point_g, take_price)
            logging.debug('Сделка закрыта по тейку')

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
            logging.debug('Сделка закрыта по стопу')
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
            logging.debug("Ни одна из цен не была достигнута.")

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
            vwap_ratio_enter
        )

        all_other_parameters_up.append(other_parameters)

        # break
        # break
    return all_other_parameters_up
