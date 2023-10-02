import numpy as np
import matplotlib.pyplot as plt

from other_modules.timing_decorator import timing_decorator
from core.point_combinations.treand_models.advanced_model_propetry import AdvancedModelProperty
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
        # down_LT_break_point = Point.find_LT_break_point_close(down.df,
        #                                                 down.t4,
        #                                                 down.properties.dist_cp_t4_x1,
        #                                                 down.LT.slope,
        #                                                 down.LT.intercept,
        #                                                 'down_model'
        #                                                 )
        # if not down_LT_break_point:
        #     continue

        LT_intersect = Point.find_intersect_two_line_point(
            up.LT.intercept,
            up.LT.slope,
            down.LT.intercept,
            down.LT.slope
        )

        # if (down.properties.target_1 > down.properties.target_3):
        #     continue
        # plt.plot(down_LT_break_point[0], down_LT_break_point[1],
        #          marker='o', color='r', markersize=10)
        #
        """------------- часть про активацию модели ------------"""

        # Находим нижний край тела каждой свечи
        lower_body_edge = up.df['close']

        # Задаем диапазон поиска
        lower_limit = int(LT_intersect[0]) + 1
        upper_limit = min(up.properties.dist_cp_t4_x2, len(up.df))

        # Находим индексы всех баров в нашем диапазоне,
        # нижний край которых закрылся выше уровня up.up_take_lines[1]
        bars_above_t4up = np.where(
            lower_body_edge[int(lower_limit):int(upper_limit)] > t4up[1])

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

        stop_price = down.t4[1]
        plt.plot(entry_index, entry_price,
                 marker='^', color='r', markersize=10)

        take_price = t4up[1] + 0.9 * (
                up.properties.up_take_100 - t4up[1])
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

        take_percent_difference = take_price * 100 / entry_price - 100

        if take_percent_difference < 1:
            continue
        # if take_percent_difference/stop_percent_difference < 0.9:
        #     continue
        # if take_percent_difference/stop_percent_difference > 1.4:
        #     continue

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

        )))

        all_setup_parameters = (base_setup_parameters, advanced_setup_parameters, up, down)
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
    all_other_parameters_up = StrategySimulator(
        'long').trade_process(setup_parameters)
    # Сохраняем результаты
    ExcelSaver(
        ticker,
        'long',
        timeframe,
        s_date,
        u_date,
        all_other_parameters_up,
    ).save()
