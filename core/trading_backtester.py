from collections import namedtuple
import pandas as pd
from core.position_evaluator_new import PositionEvaluator
import config

TradeParameters = namedtuple('TradeParameters', [
    'entry_date', 'ticker', 'entry_index', 'entry_price', 'stop_price',
    'take_price', 'stop_percent_difference', 'take_percent_difference',
])

from core.model_utilities.line import Line

import matplotlib.pyplot as plt


class StrategySimulator:
    def __init__(self, position_type='long'):
        self.position_type = position_type
        pass

    def trade_process(self, df, all_base_setup_parameters):
        """Получает входные данные сетапа,
        симулирует торговлю."""
        all_other_parameters_up = []
        for_plot = []

        for parameters in all_base_setup_parameters:

            params = TradeParameters(*parameters[0])
            model = parameters[2]

            df = df.copy()  # Создаем копию, не изменяем исходный DataFrame
            df['dateTime'] = pd.to_datetime(df['dateTime'])  # Преобразуем дату
            # Обрезаем датафрейм по интересующему нас диапазону
            #TODO вернуть 101
            close_index = params.entry_index + 101  # конечный индекс
            #TODO сделать force_close_minutes = close_index * минуты таймфрейма
            upper_limit = min(close_index, len(df))
            sub_df = df.iloc[params.entry_index:upper_limit]

            # Создаем экземпляр класса PositionEvaluator
            # Отправляем ему параметры на проторговку.
            evaluator = PositionEvaluator(sub_df,
                                          params.ticker,
                                          params.entry_date,
                                          params.take_price,
                                          params.stop_price,
                                          self.position_type,
                                          force_close_minutes=3000,
                                          )

            # Вызов метода evaluate, получаем результат
            close_position, close_reason, close_point = evaluator.evaluate()

            # Если позиция закрыта,
            # тогда определяем параметры проведенной сделки
            if close_position:
                # Определяем переменную с ценой закрытия сделки
                close_position_price = close_point[1]
                # Считаем изменение цены в % между ценой входа
                # и закрытия сделки
                if self.position_type == 'long':
                    diff = (
                            (close_position_price - params.entry_price)
                            / params.entry_price * 100
                    )
                else:
                    diff = (
                            (params.entry_price - close_position_price)
                            / params.entry_price * 100
                    )

                # Определяем была ли сделка успешной или нет
                profit_or_lose = 1 if diff > 0 else 0
                # Считаем длительность сделки
                open_to_close_trade_duration = (
                        close_point[0] - params.entry_index
                )
            else:
                close_position_price = None
                diff = None
                profit_or_lose = None
                open_to_close_trade_duration = None
                close_point = None
            # Формируем все в одну переменную, которая содержит кортеж с
            # параметрами сделки.
            result_trade = (
                # Информация о сделке
                params.entry_date,
                params.ticker,
                open_to_close_trade_duration,
                params.entry_price,
                params.stop_price,
                params.take_price,
                close_position_price,
                diff,
                profit_or_lose,
                params.stop_percent_difference,
                params.take_percent_difference,
            )
            other_parameters = result_trade + parameters[1]

            all_other_parameters_up.append(other_parameters)

            for_plot_params = (
                params.entry_index,
                params.entry_price,
                params.stop_price,
                params.take_price,
                close_point
            )
            for_plot.append(for_plot_params)

            # Если сделка была закрыта и достигла стопа - рисуем график.
            # if close_position_price is not None:
                # if profit_or_lose == 0:
            StrategySimulator.plot_trade(close_point, params, model)

        return all_other_parameters_up, for_plot

    @staticmethod
    def plot_trade(close_point, params, model):
        """Отрисовываем только модель в рамках одного сетапа."""

        from core.candle_plot import CandleStickPlotter
        # Задаем границы датафрейма для среза
        close_index = params.entry_index + 101  # конечный индекс
        upper_limit = min(close_index, len(model.df))
        sub_df_for_plot = model.df.iloc[int(model.CP[0]):upper_limit]
        # Сбрасываем индекс датафрейма
        sub_df_for_plot.reset_index(drop=True, inplace=True)

        # Начальный индекс нового DataFrame,
        # используем для пересчета индексов элементов модели и торгового сетапа
        start_index = int(model.CP[0])

        new_CP_index = int(model.CP[0] - start_index)
        # Пересчет индексов
        new_t1_index = int(model.t1[0]) - start_index
        new_t2_index = int(model.t2[0]) - start_index
        new_t3_index = int(model.t3[0]) - start_index
        new_t4_index = int(model.t4[0]) - start_index
        model.CP = (new_CP_index, model.CP[1])
        model.t1 = (new_t1_index, model.t1[1])
        model.t2 = (new_t2_index, model.t2[1])
        model.t3 = (new_t3_index, model.t3[1])
        model.t4 = (new_t4_index, model.t4[1])
        close_point_plot = ((close_point[0] - start_index),
                            close_point[1])

        entry_index_plot = (params.entry_index - start_index)
        # Формируем кортеж с параметрами торговой сделки
        for_plot_params = (
            params.ticker,
            entry_index_plot,
            params.entry_price,
            params.stop_price,
            params.take_price,
            close_point_plot
        )
        # Создаем объект для класса отрисовки
        plot = CandleStickPlotter(sub_df_for_plot)
        plot.add_candlesticks()
        # Пересчитываем лини модели
        lt_line, lc_line = StrategySimulator.lt_lc_for_plot(model)
        # Вызываем функцию отрисовки
        plot.add_trade_elements(sub_df_for_plot, model, for_plot_params, lt_line, lc_line)
        # Устанавливаем параметры сохранения изображения и сохраняем его.
        filename = (
                config.IMAGES_DIR + f'{params.ticker}-'
                                    f'{params.entry_date}.png'
        )
        plot.save(filename)

    @staticmethod
    def lt_lc_for_plot(model):
        """Пересчет линий для более удобного отображения."""

        # Задаем правую границу прямой
        right_edge_index_lc = (model.t4[0] * 2)
        right_edge_index_lt = (right_edge_index_lc
                               + (model.t4[0] - model.t3[0])
                               )
        # Пересчитываем значения прямых
        lc_line = Line.calculate_1(model.CP, model.t4, right_edge_index_lc)
        lt_line = Line.calculate_1(model.CP, model.t3, right_edge_index_lt)

        return lt_line, lc_line

    # @staticmethod
    # def lt_lc_for_plot(model):
    #
    #     x_CP = model.CP[0]
    #     y_CP = model.CP[1]
    #     print(x_CP)
    #     y_CP_LC = model.LC.slope * x_CP + model.LC.intercept
    #
    #     # Получаем индексы верхней линии (LC),
    #     # которые находятся в диапазоне x для нижней линии (LT)
    #     indices_LC = [i for i, x in enumerate(model.LC.points[0]) if
    #                   min(model.LT.points[0]) <= x <= max(
    #                       model.LT.points[0])]
    #
    #     # Получаем x и y для верхней линии (LC) в найденном диапазоне,
    #     # включая точку CP
    #     x_LC_extended = [x_CP,
    #                      *[model.LC.points[0][i] for i in indices_LC]]
    #     y_LC_extended = [y_CP_LC,
    #                      *[model.LC.points[1][i] for i in indices_LC]]
    #     x_LC_last, y_LC_last = x_LC_extended[-1], y_LC_extended[-1]
    #     lc_point2 = (x_LC_last, y_LC_last)
    #     lc_line = Line.calculate_1((x_CP, y_CP),
    #                                lc_point2)
    #
    #     # переопределяем ЛТ
    #     y_CP_LT = model.LT.slope * x_CP + model.LT.intercept
    #     # Получаем x и y для нижней линии (LT), включая точку CP
    #     x_LT_extended = [x_CP, *model.LT.points[0]]
    #     y_LT_extended = [y_CP_LT, *model.LT.points[1]]
    #     x_LT_last, y_LT_last = x_LT_extended[-1], y_LT_extended[-1]
    #     lt_point2 = (x_LT_last, y_LT_last)
    #
    #     lt_line = Line.calculate_1((x_CP, y_CP),
    #                                lt_point2)
    #
    #     return lt_line, lc_line
