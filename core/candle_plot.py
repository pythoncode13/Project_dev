import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from decimal import Decimal
from scipy.spatial import distance

from core.model_utilities.line import Line
# from core.models.up_model_property import Line, Point

PlotParameters = namedtuple('PlotParameters', ['ticker',
    'entry_index', 'entry_price', 'stop_price', 'take_price', 'close_point'
])


class CandleStickPlotter:
    """График японских свечей."""

    def __init__(self, df: pd.DataFrame):
        # self.fig, self.ax = plt.subplots(figsize=(78, 30), dpi=300)
        self.fig, self.ax = plt.subplots(figsize=(39, 15), dpi=150)
        # Задаем ширину отступа
        self.fig.tight_layout(pad=5)

        self.df = df

        self.ax.set_xlim(0, len(df))  # Устанавливаем пределы оси X от 0 до количества точек данных
    def add_candlesticks(self) -> None:
        """Построение базового графика японских свечей."""
        width = .4
        width2 = .05

        up = self.df[self.df.close >= self.df.open]
        down = self.df[self.df.close < self.df.open]

        col1 = 'green'
        col2 = 'red'

        plot_data = (
            (up, col1, 'close', 'open'),
            (down, col2, 'open', 'close'),
        )
        for direct, color, val1, val2 in plot_data:
            self.ax.bar(
                direct.index,
                direct.close - direct.open,
                width,
                bottom=direct.open,
                color=color,
            )
            self.ax.bar(
                direct.index,
                direct.high - direct[val1],
                width2, bottom=direct[val1],
                color=color,
            )
            self.ax.bar(
                direct.index,
                direct.low - direct[val2],
                width2, bottom=direct[val2],
                color=color,
            )

            # Рисуем линию по close свечам
            self.ax.plot(
                self.df.index,
                self.df['close'],
                color='blue',
                linewidth=1
            )

    def add_trend_points(self, model) -> None:

        # for model in activated_models_up:

        # Add points
        # add marker
        self.ax.plot(model.t1[0], model.t1[1], marker='o', color='k')
        self.ax.plot(model.t2[0], model.t2[1], marker='o', color='k')
        self.ax.plot(model.t3[0], model.t3[1], marker='o', color='k')
        self.ax.plot(model.t4[0], model.t4[1], marker='o', color='g')

        # add text
        self.ax.text(model.t1[0], model.t1[1], int(model.t1[0]),
                     fontsize=10)
        self.ax.text(model.t2[0], model.t2[1], 't2',
                     fontsize=10)
        self.ax.text(model.t3[0], model.t3[1], 't3',
                     fontsize=10)
        self.ax.text(model.t4[0], model.t4[1], 't4',
                     fontsize=10)
        self.ax.text(model.CP[0]-1, model.CP[1], 'CP',
                     fontsize=10)

    def add_trade_points(self, trade_plot) -> None:
        for parameters in trade_plot:

            trade = PlotParameters(*parameters)

            # Рисуем точку входа
            self.ax.plot(trade.entry_index, trade.entry_price,
                         marker='^', color='g')

            # Рисуем уровни тейка и стопа
            self.ax.hlines(
                        y=trade.take_price,
                        xmin=trade.entry_index,
                        xmax=trade.entry_index + 25,
                        colors='green',
                        linestyles='solid',
                        linewidth=0.5,
                    )

            self.ax.hlines(
                y=trade.stop_price,
                xmin=trade.entry_index,
                xmax=trade.entry_index + 25,
                colors='red',
                linestyles='solid',
                linewidth=0.5,
            )

            # Рисуем точку выхода из сделки
            # if trade.close_point[1] > trade.entry_price:
            if trade.close_point:
                self.ax.plot(trade.close_point[0], trade.close_point[1],
                             marker='^', color='r')
            # self.ax.plot(combination[0][0], combination[0][1], 'bo')

    def add_trade_elements(self, sub_df_for_plot, model, trade_plot, lt_line,
                           lc_line) -> None:
        trade = PlotParameters(*trade_plot)

        xmin = 0  # начальный индекс датафрейма (если индекс сброшен)
        xmax = len(sub_df_for_plot) - 1  # последний индекс датафрейма
        # for model in activated_models_up:

        # Добавление текста
        self.ax.text(0.01, 0.98, trade.ticker,
                     verticalalignment='top', horizontalalignment='left',
                     color='black', fontsize=40, transform=self.ax.transAxes)


        # Add points
        # add marker
        self.ax.plot(model.t1[0], model.t1[1], marker='o', color='k',
                     markersize=20)
        self.ax.plot(model.t2[0], model.t2[1], marker='o', color='k',
                     markersize=20)
        self.ax.plot(model.t3[0], model.t3[1], marker='o', color='k',
                     markersize=20)
        self.ax.plot(model.t4[0], model.t4[1], marker='o', color='g',
                     markersize=20)

        # add text
        self.ax.text(model.t1[0], model.t1[1], int(model.t1[0]),
                     fontsize=10)
        self.ax.text(model.t2[0], model.t2[1], 't2',
                     fontsize=10)
        self.ax.text(model.t3[0], model.t3[1], 't3',
                     fontsize=10)
        self.ax.text(model.t4[0], model.t4[1], 't4',
                     fontsize=10)
        self.ax.text(model.CP[0] - 1, model.CP[1], 'CP',
                     fontsize=10)
        # Рисуем прямые ЛТ и ЛЦ
        plt.plot(lt_line.points[0], lt_line.points[1], ':',
                 color='purple',
                 linewidth=0.9)

        plt.plot(lc_line.points[0], lc_line.points[1], ':',
                 color='purple',
                 linewidth=0.9)

        # for parameters in trade_plot:

        # Рисуем точку входа
        self.ax.plot(trade.entry_index, trade.entry_price,
                     marker='^', color='k', markersize=20)
        # Рисуем уровни точки входа
        self.ax.hlines(
            y=trade.entry_price,
            xmin=xmin,
            xmax=xmax,
            colors='blue',
            linestyles='solid',
            linewidth=1,
        )
        # Добавление надписи
        self.ax.text(xmax, trade.entry_price, 'Entry',
                     verticalalignment='bottom', horizontalalignment='right',
                     color='black', fontsize=40)

        # Рисуем уровни тейка и стопа
        self.ax.hlines(
            y=trade.take_price,
            xmin=xmin,
            xmax=xmax,
            colors='green',
            linestyles='solid',
            linewidth=1,
        )
        # Добавление надписи
        self.ax.text(xmax, trade.take_price, 'Take Price 2',
                     verticalalignment='bottom', horizontalalignment='right',
                     color='black', fontsize=40)

        # Рисуем уровни тейка и стопа
        self.ax.hlines(
            y=model.properties.up_take_100,
            xmin=xmin,
            xmax=xmax,
            colors='green',
            linestyles='solid',
            linewidth=1,
        )

        # Добавление надписи
        self.ax.text(xmax, model.properties.up_take_100, 'Take Price 1',
                     verticalalignment='bottom', horizontalalignment='right',
                     color='black', fontsize=40)

        self.ax.hlines(
            y=trade.stop_price,
            xmin=xmin,
            xmax=xmax,
            colors='red',
            linestyles='solid',
            linewidth=1,
        )
        # Добавление надписи
        self.ax.text(xmax, trade.stop_price, 'Stop',
                     verticalalignment='bottom', horizontalalignment='right',
                     color='black', fontsize=40)

        # Рисуем точку выхода из сделки, при ее наличии
        # if trade.close_point[1] > trade.entry_price:
        if trade.close_point:
            self.ax.plot(trade.close_point[0], trade.close_point[1],
                         marker='^', color='k', markersize=10)

            # t2 = point[1]
            # t3 = point[2]
            # t4 = point[3]
            # LT = point[4]
            # t5up = point[4]

            # self.ax.plot(t1[0], t1[1], 'bo')
            # self.ax.text(t1up[0], t1up[1], t1up[0], fontsize=10)
            # self.ax.plot(t2up[0], t2up[1], 'o', color='k')
            # self.ax.text(t2up[0], t2up[1], point[5], fontsize=10)
            # self.ax.plot(t3up[0], t3up[1], 'ro')
            # self.ax.plot(t4up, t2up[1], marker='^', color='g')
            # plt.plot(LT.points[0], LT.points[1], ':', color='purple',
            #              linewidth=0.9)
            #
            # self.ax.hlines(
            #             y=point.up_take_lines[1],
            #             xmin=t1up[0],
            #             xmax=t3up[0] + 25,
            #             colors='green',
            #             linestyles='solid',
            #             linewidth=0.5,
            #         )
            # self.ax.text(t4up[0], t4up[1]+50, t4up[0], fontsize=10)
            # if t5up:
            #     self.ax.plot(t5up[0], t5up[1], color='red',
            #              marker='>',
            #              markersize=3, markeredgecolor='black')

    # def add_trend_points(self, all_t1_up, all_t2_up, all_t3_up, all_t4_up, combinations, new_combinations, up_trend_points_test) -> None:
        # for point in all_t1_up:
        #     self.ax.plot(point[0], point[1], 'bo')

        # for point in all_t2_up:
        #     self.ax.plot(point[0], point[1], 'o', color='k')

        # for point in all_t3_up:
        #     self.ax.plot(point[0], point[1], 'ro')
        #
        # for point in all_t4_up:
        #     self.ax.plot(point[0], point[1], 'go')
        #     plt.text(point[0], point[1], point[0], fontsize=10)

        # for combination in combinations:
        #     # self.ax.plot(combination[0][0], combination[0][1], 'bo')
        #     t1up = combination[0]
        #     t2up = combination[1]
        #     t3up = combination[2]
        #
        #     # проводим линию ЛЦ
        #     slope_LT, intercept_LT, LT_up = Line.calculate(t1up, t3up)
        #
        #     plt.plot(LT_up[0], LT_up[1], ':', color='purple',
        #              linewidth=0.9)
        #     # self.ax.plot(t1up[0], t1up[1], 'bo')
        #     self.ax.plot(t2up[0], t2up[1], '>', color='r')
        #     self.ax.plot(t3up[0], t3up[1], 'ro')

        # for combination in new_combinations:
        #     # self.ax.plot(combination[0][0], combination[0][1], 'bo')
        #     t1up = combination[0]
        #     t2up = combination[1]
        #     t3up = combination[2]
        #     t4up = combination[3]
        #
        #     # проводим линию ЛЦ
        #     slope_LT, intercept_LT, LT_up = Line.calculate(t1up, t3up)
        #
        #     # проводим линию ЛЦ
        #     slope_LC, intercept_LC, LC_up = Line.calculate(t2up, t4up)
        #
        #     plt.plot(LT_up[0], LT_up[1], ':', color='purple',
        #              linewidth=0.9)
        #     plt.plot(LC_up[0], LC_up[1], ':', color='purple',
        #              linewidth=0.9)
        #     self.ax.plot(t1up[0], t1up[1], 'bo')
        #     self.ax.plot(t2up[0], t2up[1], 'o', color='k')
        #     self.ax.plot(t3up[0], t3up[1], 'ro')
        #     self.ax.plot(t4up[0], t4up[1], 'go')

        # for point in up_trend_points_test:
        #
        #     # self.ax.plot(combination[0][0], combination[0][1], 'bo')
        #     t1up = point[0]
        #     t2up = point[1]
        #     t3up = point[2]
        #     t4up = point[3]
        #     t5up = point[4]
        #
        #     # self.ax.plot(t1up[0], t1up[1], 'bo')
        #     self.ax.text(t1up[0], t1up[1], t1up[0], fontsize=10)
        #     # self.ax.plot(t2up[0], t2up[1], 'o', color='k')
        #     self.ax.plot(t3up[0], t3up[1], 'ro')
        #     self.ax.plot(t4up[0], t4up[1], 'go')
        #     self.ax.text(t4up[0], t4up[1]+50, t4up[0], fontsize=10)
        #     if t5up:
        #         self.ax.plot(t5up[0], t5up[1], color='red',
        #                  marker='>',
        #                  markersize=3, markeredgecolor='black')

    # def add_trend_points(self, trend_points) -> None:
    #     """Добавление трендовых точек на график."""
    #     for t1down, t2down, t3down in trend_points:
    #         self.ax.plot(t1down[0], t1down[1], 'yo')
    #         self.ax.plot(t2down[0], t2down[1], 'bo')
    #         self.ax.plot(t3down[0], t3down[1], 'ro')
    #
    #         # рисуем горизонтальную линию через т2
    #         self.ax.hlines(
    #             y=t2down[1],
    #             xmin=t2down[0],
    #             xmax=t2down[0] + 25,
    #             colors='green',
    #             linestyles='solid',
    #             linewidth=0.5,
    #         )
    #
    #         # добавляем текст ценового значения
    #         self.ax.text(
    #             t2down[0],
    #             t2down[1],
    #             f't2down_line: {t2down[1]:.2f}',
    #             color='black',
    #             fontsize=7,
    #             ha='center',
    #             va='bottom',
    #         )
    #
    #         # рисуем горизонтальную линию через т1
    #         self.ax.hlines(
    #             y=t1down[1],
    #             xmin=t1down[0],
    #             xmax=t1down[0] + 50,
    #             colors='r',
    #             linestyles='solid',
    #             linewidth=0.5,
    #         )
    #         # добавляем текст "Stop" и ценовое значение
    #         self.ax.text(
    #             t1down[0],
    #             t1down[1],
    #             f'STOP: {t1down[1]:.2f}',
    #             color='black',
    #             fontsize=7,
    #             ha='center',
    #             va='bottom',
    #         )

    def show(self) -> None:
        """Показать график."""
        self.ax.show()

    def save(self, filename) -> None:
        """Сохранить график."""
        self.fig.savefig(filename)
