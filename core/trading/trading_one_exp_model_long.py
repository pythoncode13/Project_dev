import numpy as np

from core.models_trade_property.base_trade_analysis_long import \
    BaseTradeAnalysis
from other_modules.timing_decorator import timing_decorator
from core.point_combinations.treand_models.advanced_model_propetry import AdvancedModelProperty
from core.trading_backtester import StrategySimulator
from utils.excel_saver import ExcelSaver
from core.model_utilities.line import Line


def prepare_trading_setup(activated_models_up, ticker):
    """Подготовка данных для сетапа."""

    all_base_setup_parameters = []

    for model in activated_models_up:
        t1 = model.t1
        t2 = model.t2
        t3 = model.t3
        t4 = model.t4
        # Сила модели
        if (int(t3[0]) - int(t1[0])) < (int(t1[0]) - int(model.CP[0])):
            continue
        if Line.check_line(model.df,
                           model.LC.slope,
                           model.LC.intercept,
                           (t1[0] + 1, 0), t4,
                           direction='close'):
            continue
        """------------- часть про активацию модели ------------"""
        # Находим нижний край тела каждой свечи
        lower_body_edge = model.df[['high']].min(axis=1)

        # Задаем диапазон поиска
        lower_limit = int(t4[0] + 1)
        upper_limit = min(int(model.properties.dist_cp_t4_x1), len(model.df))

        # Находим индексы всех баров в нашем диапазоне,
        # нижний край которых закрылся выше уровня model.up_take_lines[1]
        bars_above_t4 = np.where(
            lower_body_edge[lower_limit:upper_limit] > t4[1])

        # Если таких баров нет, bars_above_t4 будет пустым
        # и мы пропускаем этот шаг цикла
        if bars_above_t4[0].size == 0:
            continue

        # В противном случае, берем индекс первого такого бара,
        # учитывая смещение на lower_limit
        first_bar_above_t4 = bars_above_t4[0][0] + lower_limit
        """------------- формирование сетапа ------------"""
        x_CP = float(model.CP[0])
        lower_limit = max(0, int(x_CP - (t4[0] - x_CP) * 3))
        upper_limit = min(int(first_bar_above_t4 + 110), len(model.df))
        sliced_df = model.df.iloc[lower_limit:upper_limit]

        advanced_property = AdvancedModelProperty(model, sliced_df)
        _, _, mse_t3_t4, _, _ = advanced_property.get_mse

        if mse_t3_t4 < 0.0001:
            continue

        entry_index = first_bar_above_t4
        entry_date = sliced_df.loc[entry_index, 'dateTime']

        entry_price = t4[1]

        stop_price = t1[1]

        take_price = t4[1] + 0.9 * (
                model.properties.take_200 - t4[1])

        analysis_base = BaseTradeAnalysis(t1,
                                          t2,
                                          t3,
                                          entry_price,
                                          stop_price,
                                          take_price)

        base_setup_parameters = (
            entry_date,
            ticker,
            entry_index,
            entry_price,
            stop_price,
            take_price,
            np.round(analysis_base.stop_percent_difference, 4),
            np.round(analysis_base.take_percent_difference, 4),
        )

        advanced_setup_parameters = tuple(map(lambda x: np.round(float(x), 4),
                                              (
            # model.parallel,
        )))

        all_setup_parameters = (base_setup_parameters, advanced_setup_parameters, model)
        all_base_setup_parameters.append(all_setup_parameters)

    return all_base_setup_parameters


def trade_one_exp_model_long(models, ticker, timeframe, s_date, u_date):
    # Отбираем из модели для торговли
    setup_parameters = prepare_trading_setup(models, ticker)
    # Торгуем выбранные модели
    all_other_parameters_up = StrategySimulator('close',
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
