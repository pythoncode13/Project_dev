import numpy as np
import matplotlib.pyplot as plt

from other_modules.timing_decorator import timing_decorator
from core.point_combinations.treand_models.advanced_model_propetry import \
    AdvancedModelProperty
from core.model_utilities.point import Point
from core.trading_backtester import StrategySimulator
from utils.excel_saver import ExcelSaver
from core.trading.two_models_2.two_models import TwoModel


@timing_decorator
def prepare_trading_setup(super_groups, ticker):
    """Подготовка данных для сетапа."""

    all_base_setup_parameters = []

    for two_model_group in super_groups:
        up = two_model_group.up_model
        down = two_model_group.down_model
        t1up = up.t1
        t2up = up.t2
        t3up = up.t3
        t4up = up.t4
        """Проверяем, что работаем с пентаграммой."""
        if down.CP[0] > up.t1[0]:
            continue
        """ 
        Проверяем, является ли 1 модель (ап) моделью по тренду.
        Если да, тогда продолжаем цикл.
        """
        # т1ап - не мин лоу на участке ст-4х2 влево
        dist_cp_t4_up = int(up.t4[0]) - int(up.CP[0])
        dist_cp_t4_x2_up_left = int(up.CP[0]) - dist_cp_t4_up * 2
        # Получаем все лоу на участке dist_cp_t4_x2_up_left:up.t1[0] - 1
        low_values = up.df.loc[dist_cp_t4_x2_up_left:up.t1[0] - 1]['low']

        # Пропустить цикл, если все лоу выше или равны up.t1[1]
        if all(low > up.t1[1] for low in low_values):
            continue
        # Получаем все хай на участке dist_cp_t4_x2_up_left:down.t1[0] - 1
        high_values = up.df.loc[dist_cp_t4_x2_up_left:down.t1[0] - 1]['high']

        # Пропустить цикл, если есть хоть один хай выше или равны down.t1[1]
        if any(high >= down.t1[1] for high in high_values):
            continue

        plt.plot(up.t1[0], up.t1[1],
                 marker='^', color='b', markersize=10)
        plt.plot(down.t1[0], down.t1[1],
                 marker='^', color='r', markersize=10)
        # down_LT_break_point = Point.find_line_break_point_close(down.df,
        #                                                 down.t4[0],
        #                                                 down.properties.dist_cp_t4_x1,
        #                                                 down.LT.slope,
        #                                                 down.LT.intercept,
        #                                                 'above'
        #                                                 )
        # if not down_LT_break_point:
        #     continue
        up_LT_break_point = Point.find_line_break_point_close(up.df,
                                                              up.t4[0],
                                                              up.properties.dist_cp_t4_x1,
                                                              up.LT.slope,
                                                              up.LT.intercept,
                                                              'below'
                                                              )
        if not up_LT_break_point:
            continue
        # LT_intersect = Point.find_intersect_two_line_point(
        #     up.LT.intercept,
        #     up.LT.slope,
        #     down.LT.intercept,
        #     down.LT.slope
        # )

        # if (down.properties.target_1 > down.properties.target_3):
        #     continue
        plt.plot(up_LT_break_point[0], up_LT_break_point[1],
                 marker='o', color='r', markersize=10)
        #
        """------------- часть про активацию модели ------------"""
        # down_LT_break_point = Point.find_LT_break_point(down.df,
        #                                                       down.t4,
        #                                                       down.properties.dist_cp_t4_x1,
        #                                                       down.LT.slope,
        #                                                       down.LT.intercept,
        #                                                       'down_model'
        #                                                       )
        # if not down_LT_break_point:
        #     continue
        # Находим нижний край тела каждой свечи
        lower_body_edge = up.df['close']

        # Задаем диапазон поиска
        lower_limit = int(up_LT_break_point[0]) + 1
        upper_limit = min(up.properties.dist_cp_t4_x2, len(up.df))

        # Находим индексы всех баров в нашем диапазоне,
        # нижний край которых закрылся выше уровня up.up_take_lines[1]
        bars_above_t4up = np.where(
            lower_body_edge[int(lower_limit):int(upper_limit)] < down.t4[1])

        # Если таких баров нет, bars_above_t4up будет пустым
        # и мы пропускаем этот шаг цикла
        if bars_above_t4up[0].size == 0:
            continue

        # В противном случае, берем индекс первого такого бара,
        # учитывая смещение на lower_limit
        first_bar_above_t4up = bars_above_t4up[0][0] + lower_limit
        # first_bar_above_t4up = int(down_LT_break_point[0])
        """------------- формирование сетапа ------------"""
        x_CP = float(up.CP[0])
        lower_limit = max(0, int(x_CP - (t4up[0] - x_CP) * 3))
        upper_limit = min(int(first_bar_above_t4up) + 110, len(up.df))
        sliced_df = up.df.iloc[lower_limit:upper_limit]

        advanced_property = AdvancedModelProperty(up, sliced_df)
        # _, _, mse_t3_t4, _, _ = advanced_property.get_mse

        # if mse_t3_t4 < 0.0001:
        #     continue

        entry_index = int(first_bar_above_t4up)
        entry_date = sliced_df.loc[entry_index, 'dateTime']

        entry_price = sliced_df.loc[entry_index, 'close']
        # entry_price = down.t4[1]

        stop_price = up.t4[1]
        plt.plot(entry_index, entry_price,
                 marker='^', color='r', markersize=10)

        take_price = down.t4[1] - 0.9 * (
                down.t4[1] - down.properties.take_200)
        # take_price = down.properties.target_3
        # take_price = entry_price + 0.9 * (
        #         down.properties.target_1 - entry_price)
        print(take_price)
        # min_low = up.df['low'].iloc[int(up.t4[0]):int(entry_index)].min()
        # if up.properties.target_1 < min_low:
        #     continue

        # if (up.df['close'][int(down.t4[0])+1:int(entry_index)-1] <= stop_price).any():
        #     # df.loc[int(LT_break_point[0]), 'close']
        #     continue

        # analysis_base = BaseTradeAnalysis(t1up,
        #                                   t2up,
        #                                   t3up,
        #                                   entry_price,
        #                                   stop_price,
        #                                   take_price)

        stop_percent_difference = 100 - stop_price * 100 / entry_price

        take_percent_difference = (take_price * 100 / entry_price - 100) * -1
        take_percent_difference1 = (
                                               down.properties.take_100 * 100 / entry_price - 100) * -1
        # plt.hlines(
        #     y=down.properties.take_100,
        #     xmin=down.t1[0],
        #     xmax=len(down.df),
        #     colors='green',
        #     linestyles='solid',
        #     linewidth=1,
        # )
        if take_percent_difference1 < 1:
            continue
        # if take_percent_difference/stop_percent_difference < 0.9:
        #     continue
        # if take_percent_difference/stop_percent_difference > 1.4:
        #     continue
        strong_t1_cp = (down.t1[0] - int(down.CP[0]))
        strong_t3_t1 = (down.t3[0] - down.t1[0]) * 3
        base_setup_parameters = (
            entry_date,
            ticker,
            entry_index,
            entry_price,
            stop_price,
            take_price,
            np.round(stop_percent_difference, 4),
            np.round(take_percent_difference, 4),
        )

        advanced_setup_parameters = tuple(map(lambda x: np.round(float(x), 4),
                                              (
                                                  strong_t1_cp,
                                                  strong_t3_t1
                                              )))

        all_setup_parameters = (
            base_setup_parameters, advanced_setup_parameters, up, down)
        all_base_setup_parameters.append(all_setup_parameters)

    return all_base_setup_parameters


def trade_two_exp_model_long(candidates_up,
                             candidates_down,
                             ticker,
                             timeframe,
                             s_date,
                             u_date
                             ):
    # Объединяем две модели
    super_groups = TwoModel(
        candidates_up,
        candidates_down
    ).find_two_model()

    # Отбираем пары моделей для торговли
    setup_parameters = prepare_trading_setup(super_groups, ticker)
    # Торгуем выбранные пары
    all_other_parameters_up = StrategySimulator('close',
                                                'short').trade_process(
        setup_parameters)
    # Сохраняем результаты
    ExcelSaver(
        ticker,
        'short',
        timeframe,
        s_date,
        u_date,
        all_other_parameters_up,
    ).save()
