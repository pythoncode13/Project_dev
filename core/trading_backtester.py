"""Симулятор стратегии."""

import pandas as pd
from collections import namedtuple
import config
from core.candle_plot import CandleStickPlotter
from core.position_evaluator_new import PositionEvaluator

# Named tuple для удобства хранения параметров торговли
TradeParameters = namedtuple(
    'TradeParameters',
    ['entry_date', 'ticker', 'entry_index', 'entry_price', 'stop_price',
     'take_price', 'stop_percent_difference', 'take_percent_difference']
)


class StrategySimulator:
    def __init__(self, position_type='long'):
        """Инициализация симулятора стратегии."""
        self.position_type = position_type

    def trade_result(self, params, close_point):
        """
        Расчет результатов торговли.

        :param params: параметры сделки
        :param close_point: точка закрытия позиции
        :return: результаты сделки
        """
        # Если PositionEvaluator вернула пустой close_point - возвращаем None.
        if not close_point:
            return None, None, None, None
        # Определяем переменную с ценой закрытия сделки
        close_position_price = close_point[1]
        # Считаем изменение цены в % между ценой входа
        # и закрытия сделки
        price_diff = close_position_price - params.entry_price
        diff = (
            price_diff / params.entry_price * 100
            if self.position_type == 'long'
            else -price_diff / params.entry_price * 100
        )

        return (
                close_position_price,
                diff,
                int(diff > 0), #profit_or_lose
                close_point[0] - params.entry_index # длительность сделки
                )

    def trade_process(self, all_base_setup_parameters):
        """
        Процесс симуляции торговли.

        :param df: исходный DataFrame
        :param all_base_setup_parameters: параметры всех сетапов
        :return: результаты всех сделок
        """
        all_other_parameters_up = []

        for parameters in all_base_setup_parameters:
            params = TradeParameters(*parameters[0])
            model = parameters[2]

            # Срез данных для анализа
            sub_df = model.df.iloc[
                params.entry_index: min(
                    params.entry_index + 101, len(model.df)
                )
            ].copy()
            sub_df['dateTime'] = pd.to_datetime(sub_df['dateTime'])

            # Проводим виртуальную торговлю в PositionEvaluator
            close_position, close_reason, close_point = PositionEvaluator(
                sub_df,
                params.ticker,
                params.entry_date,
                params.take_price,
                params.stop_price,
                self.position_type,
                force_close_minutes=3000
            ).evaluate()

            # Расчет результатов
            (
                close_position_price,
                diff,
                profit_or_lose,
                trade_duration
            ) = self.trade_result(params, close_point)

            # Собираем все параметры в одну переменную
            all_other_parameters_up.append(
                (
                    params.entry_date, params.ticker, trade_duration,
                    params.entry_price, params.stop_price, params.take_price,
                    close_position_price, diff, profit_or_lose,
                    params.stop_percent_difference, params.take_percent_difference
                ) + parameters[1]
            )

            # Отрисовываем сделку
            self.plot_trade(close_point, params, model)

        return all_other_parameters_up

    def plot_trade(self, close_point, params, model):
        """
        Отрисовка результатов сделки.

        :param close_point: точка закрытия
        :param params: параметры сделки
        :param model: модель
        """
        # Задаем границы датафрейма для среза
        start_index = int(model.CP[0])
        # Инициализируем initialize_plot для пересчета точек
        # с учетом обновленного индекса
        model.initialize_plot(start_index)
        # min(params.entry_index + 101, len(model.df)) - конечный индекс
        sub_df_for_plot = model.df.iloc[
            start_index: min(params.entry_index + 101, len(model.df))
        ]
        # Сбрасываем индекс датафрейма
        sub_df_for_plot.reset_index(drop=True, inplace=True)

        # Если есть close_point_plot - пересчитываем индекс
        close_point_plot = (
            (close_point[0] - start_index), close_point[1]
        ) if close_point else None
        # И для точки вода тоже пересчитываем
        entry_index_plot = params.entry_index - start_index

        # Создаем объект для класса отрисовки
        plot = CandleStickPlotter(sub_df_for_plot)
        plot.add_candlesticks()
        # Вызываем функцию отрисовки
        plot.add_trade_elements(
            sub_df_for_plot, model.plot,
            (
                params.ticker, entry_index_plot, params.entry_price,
                params.stop_price, params.take_price, close_point_plot
            )
        )
        plot.save(f"{config.IMAGES_DIR}"
                  f"{params.ticker}-"
                  f"{params.entry_date}.png")





# import pandas as pd
# from collections import namedtuple
#
# import config
# from core.candle_plot import CandleStickPlotter
# from core.position_evaluator_new import PositionEvaluator
#
# TradeParameters = namedtuple('TradeParameters', [
#     'entry_date', 'ticker', 'entry_index', 'entry_price', 'stop_price',
#     'take_price', 'stop_percent_difference', 'take_percent_difference',
# ])
#
#
# class StrategySimulator:
#     def __init__(self, position_type='long'):
#         self.position_type = position_type
#         pass
#

    # @staticmethod
    # def trade_result(self, params, close_point):
    #     # Анализ результатов PositionEvaluator
    #     # Если позиция закрыта,
    #     # тогда определяем параметры проведенной сделки
    #     if close_point:
    #         # Определяем переменную с ценой закрытия сделки
    #         close_position_price = close_point[1]
    #         # Считаем изменение цены в % между ценой входа
    #         # и закрытия сделки
    #         if self.position_type == 'long':
    #             diff = (
    #                     (close_position_price - params.entry_price)
    #                     / params.entry_price * 100
    #             )
    #         else:
    #             diff = (
    #                     (params.entry_price - close_position_price)
    #                     / params.entry_price * 100
    #             )
    #
    #         # Определяем была ли сделка успешной или нет
    #         profit_or_lose = 1 if diff > 0 else 0
    #         # Считаем длительность сделки
    #         open_to_close_trade_duration = (
    #                 close_point[0] - params.entry_index
    #         )
    #     else:
    #         close_position_price = None
    #         diff = None
    #         profit_or_lose = None
    #         open_to_close_trade_duration = None
    #     return (
    #             close_position_price,
    #             diff, profit_or_lose,
    #             open_to_close_trade_duration
    #     )

    # def trade_process(self, df, all_base_setup_parameters):
    #     """Получает входные данные сетапа,
    #     симулирует торговлю."""
    #     all_other_parameters_up = []
    #
    #     for parameters in all_base_setup_parameters:
    #         params = TradeParameters(*parameters[0])
    #         model = parameters[2]
    #
    #         df = df.copy()  # Создаем копию, не изменяем исходный DataFrame
    #         df['dateTime'] = pd.to_datetime(df['dateTime'])  # Преобразуем дату
    #         # Обрезаем датафрейм по интересующему нас диапазону
    #         close_index = params.entry_index + 101  # конечный индекс
    #         upper_limit = min(close_index, len(df))
    #         sub_df = df.iloc[params.entry_index:upper_limit]
    #
    #         # Проводим виртуальную торговлю в PositionEvaluator.
    #         close_position, close_reason, close_point = PositionEvaluator(
    #             sub_df,
    #             params.ticker,
    #             params.entry_date,
    #             params.take_price,
    #             params.stop_price,
    #             self.position_type,
    #             force_close_minutes=3000,
    #         ).evaluate()
    #
    #         # Оценка результатов торговли
    #         (
    #             close_position_price,
    #             diff, profit_or_lose,
    #             open_to_close_trade_duration
    #         ) = self.trade_result(params, close_point)
    #
    #         # Формируем все в одну переменную, которая содержит кортеж с
    #         # параметрами сделки.
    #         result_trade = (
    #             # Информация о сделке
    #             params.entry_date,
    #             params.ticker,
    #             open_to_close_trade_duration,
    #             params.entry_price,
    #             params.stop_price,
    #             params.take_price,
    #             close_position_price,
    #             diff,
    #             profit_or_lose,
    #             params.stop_percent_difference,
    #             params.take_percent_difference,
    #         )
    #         other_parameters = result_trade + parameters[1]
    #
    #         all_other_parameters_up.append(other_parameters)
    #
    #         StrategySimulator.plot_trade(close_point, params, model)
    #
    #     return all_other_parameters_up
    # def trade_process(self, df, all_base_setup_parameters):
    #     all_other_parameters_up = []
    #     for parameters in all_base_setup_parameters:
    #         params = TradeParameters(*parameters[0])
    #         model = parameters[2]
    #
    #         sub_df = df.iloc[params.entry_index: min(params.entry_index + 101,
    #                                                  len(df))].copy()
    #         sub_df['dateTime'] = pd.to_datetime(sub_df['dateTime'])
    #
    #         close_position, close_reason, close_point = PositionEvaluator(
    #             sub_df, params.ticker, params.entry_date,
    #             params.take_price, params.stop_price,
    #             self.position_type, force_close_minutes=3000
    #         ).evaluate()
    #
    #         close_position_price, diff, profit_or_lose, trade_duration = self.trade_result(
    #             params, close_point)
    #         all_other_parameters_up.append((
    #                                            params.entry_date,
    #                                            params.ticker, trade_duration,
    #                                            params.entry_price,
    #                                            params.stop_price,
    #                                            params.take_price,
    #                                            close_position_price, diff,
    #                                            profit_or_lose,
    #                                            params.stop_percent_difference,
    #                                            params.take_percent_difference
    #                                        ) + parameters[1])
    #
    #         self.plot_trade(close_point, params, model)
    #
    #     return all_other_parameters_up
    # @staticmethod
    # def plot_trade(close_point, params, model):
    #     """Отрисовываем только модель в рамках одного сетапа."""
    #
    #     # Задаем границы датафрейма для среза
    #     close_index = params.entry_index + 101  # конечный индекс
    #     upper_limit = min(close_index, len(model.df))
    #     sub_df_for_plot = model.df.iloc[int(model.CP[0]):upper_limit]
    #     # Сбрасываем индекс датафрейма
    #     sub_df_for_plot.reset_index(drop=True, inplace=True)
    #
    #     # Начальный индекс нового DataFrame,
    #     # используем для пересчета индексов элементов модели и торгового сетапа
    #     start_index = int(model.CP[0])
    #
    #     model.initialize_plot(start_index)
    #
    #     if close_point is not None:
    #         close_point_plot = ((close_point[0] - start_index),
    #                             close_point[1])
    #     else:
    #         close_point_plot = None
    #
    #     entry_index_plot = (params.entry_index - start_index)
    #     # Формируем кортеж с параметрами торговой сделки
    #     for_plot_params = (
    #         params.ticker,
    #         entry_index_plot,
    #         params.entry_price,
    #         params.stop_price,
    #         params.take_price,
    #         close_point_plot
    #     )
    #
    #     # Создаем объект для класса отрисовки
    #     plot = CandleStickPlotter(sub_df_for_plot)
    #     plot.add_candlesticks()
    #     # Вызываем функцию отрисовки
    #     plot.add_trade_elements(sub_df_for_plot, model.plot, for_plot_params)
    #
    #     # Устанавливаем параметры сохранения изображения и сохраняем его.
    #     filename = (
    #             config.IMAGES_DIR + f'{params.ticker}-'
    #                                 f'{params.entry_date}.png'
    #     )
    #     plot.save(filename)

# for_plot_params = (
#     params.entry_index,
#     params.entry_price,
#     params.stop_price,
#     params.take_price,
#     close_point
# )
# for_plot.append(for_plot_params)

# Если сделка была закрыта и достигла стопа - рисуем график.
# if close_position_price is not None:
# if profit_or_lose == 0:

