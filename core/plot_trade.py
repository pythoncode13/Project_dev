from core.candle_plot import CandleStickPlotter

import config
from core.model_utilities.line import Line


class PlotTrade:
    def __init__(self):
        pass

    #     for_plot_params = (
    #         params.entry_index,
    #         params.entry_price,
    #         params.stop_price,
    #         params.take_price,
    #         close_point
    #     )
    #     for_plot.append(for_plot_params)
    #
    #     # Если сделка была закрыта и достигла стопа - рисуем график.
    #     # if close_position_price is not None:
    #     # if profit_or_lose == 0:
    #     PlotTrade.plot_trade(close_point, params, model)

    @staticmethod
    def plot_trade(close_point, params, model):
        """Отрисовываем только модель в рамках одного сетапа."""

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
        lt_line, lc_line = PlotTrade.lt_lc_for_plot(model)
        # Вызываем функцию отрисовки
        plot.add_trade_elements(sub_df_for_plot, model, for_plot_params, lt_line,
                                lc_line)
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