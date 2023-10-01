"""Симулятор торговли."""

import pandas as pd
from collections import namedtuple
import config
from core.candle_plot import CandleStickPlotter
from core.position_evaluator_new import PositionEvaluator
import matplotlib.pyplot as plt
from core.model_utilities.point import Point

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
            model2 = parameters[3]
            # Срез данных для анализа
            sub_df = model.df.iloc[
                params.entry_index: min(
                    params.entry_index + 101, len(model.df)
                )
            ].copy()
            sub_df['dateTime'] = pd.to_datetime(sub_df['dateTime'], format='%Y-%m-%d %H-%M-%S')


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
            # self.plot_trade(close_point, params, model, model2)
            if close_position_price is not None:
                if profit_or_lose == 0:
                    self.plot_trade(close_point, params, model, model2, 'stop')

                else:
                    self.plot_trade(close_point, params, model, model2,
                                                 'take')
        return all_other_parameters_up

    def plot_trade(self, close_point, params, model, model2, direction):
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
        LT_intersect = None
        plot.add_trade_elements(
            sub_df_for_plot, model.plot,
            (
                params.ticker, entry_index_plot, params.entry_price,
                params.stop_price, params.take_price, close_point_plot
            ),
            LT_intersect
        )
        if model2:
            LT_intersect = Point.find_intersect_two_line_point(
                model.LT.intercept,
                model.LT.slope,
                model2.LT.intercept,
                model2.LT.slope
            )
            LT_intersect = (float(LT_intersect[0]) - start_index, LT_intersect[1])
            print(LT_intersect)
            model2.initialize_plot(start_index)
            plot.add_trade_elements(
            sub_df_for_plot, model2.plot,
            (
                params.ticker, entry_index_plot, params.entry_price,
                params.stop_price, params.take_price, close_point_plot
                ),
                LT_intersect
            )


        # plot.save(f"{config.IMAGES_DIR}"
        #           f"{params.ticker}-"
        #           f"{params.entry_date}.png")

        if direction == 'stop':
            plot.save(
                    f'{config.IMAGES_DIR}'
                    f'STOP_{params.ticker}-'
                    f'{params.entry_date}.png'
                    )
        else:
            plot.save(
                    f'{config.IMAGES_DIR}'
                    f'TAKE_{params.ticker}-'
                    f'{params.entry_date}.png'
                    )
