import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from core.model_utilities.point import Point
from core.model_utilities.line import Line

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

        # Устанавливаем пределы оси X от 0 до количества точек данных
        self.ax.set_xlim(0, len(df))

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

    def add_model_points(self, model) -> None:
        """Добавление точек модели на график."""

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

    @staticmethod
    def add_model_line(line):
        """Добавление прямых."""
        plt.plot(line.points[0], line.points[1], ':',
                 color='purple',
                 linewidth=0.9)

    def add_trade_lines(self, model, trade, xmin, xmax):
        """Добавление элементов трейда: уровни, точки входа/тейка/стопа."""

        # Рисуем точку входа
        self.ax.plot(trade.entry_index, trade.entry_price,
                     marker='^', color='k', markersize=20)
        # Рисуем уровень
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

        # Тейк
        # Прямая
        self.ax.hlines(
            y=model.properties.take_100,
            xmin=xmin,
            xmax=xmax,
            colors='green',
            linestyles='solid',
            linewidth=1,
        )

        # Надпись
        self.ax.text(xmax, model.properties.take_100, 'Take Price 1',
                     verticalalignment='bottom', horizontalalignment='right',
                     color='black', fontsize=40)
        # Прямая
        self.ax.hlines(
            y=trade.take_price,
            xmin=xmin,
            xmax=xmax,
            colors='green',
            linestyles='solid',
            linewidth=1,
        )
        # Надпись
        self.ax.text(xmax, trade.take_price, 'Take Price 2',
                     verticalalignment='bottom', horizontalalignment='right',
                     color='black', fontsize=40)

        # Стоп
        # Прямая
        self.ax.hlines(
            y=trade.stop_price,
            xmin=xmin,
            xmax=xmax,
            colors='red',
            linestyles='solid',
            linewidth=1,
        )
        # Надпись
        self.ax.text(xmax, trade.stop_price, 'Stop',
                     verticalalignment='bottom', horizontalalignment='right',
                     color='black', fontsize=40)

        # Если сделка закрыта - рисуем точку закрытия
        if trade.close_point:
            self.ax.plot(trade.close_point[0], trade.close_point[1],
                         marker='^', color='k', markersize=10)

        # Если есть пробой ЛТ - рисуем таргеты
        # Таргет 1
        if model.properties.target_1:
            print(model.properties.target_1)
            # Прямая
            self.ax.hlines(
                y=model.properties.target_1,
                xmin=model.LT_break_point[0],
                xmax=model.LT_break_point[0]+30,
                colors='black',
                linestyles='solid',
                linewidth=1,
            )
            # Надпись
            self.ax.text(model.LT_break_point[0]+5, model.properties.target_1, 'Target 1',
                         verticalalignment='bottom',
                         horizontalalignment='left',
                         color='black', fontsize=30)

            # Таргет 3
            # Прямая
            self.ax.hlines(
                y=model.properties.target_3,
                xmin=model.LT_break_point[0],
                xmax=model.LT_break_point[0]+30,
                colors='black',
                linestyles='solid',
                linewidth=1,
            )
            # Надпись
            self.ax.text(model.LT_break_point[0] + 5,
                         model.properties.target_3, 'Target 3',
                         verticalalignment='bottom',
                         horizontalalignment='left',
                         color='black', fontsize=30)

            # Таргет 5
            # Прямая
            self.ax.hlines(
                y=model.properties.target_5,
                xmin=model.LT_break_point[0],
                xmax=model.LT_break_point[0]+30,
                colors='black',
                linestyles='solid',
                linewidth=1,
            )
            # Надпись
            self.ax.text(model.LT_break_point[0] + 5,
                         model.properties.target_5, 'Target 5',
                         verticalalignment='bottom',
                         horizontalalignment='left',
                         color='black', fontsize=30)

    def add_trade_elements(self, sub_df_for_plot, model, trade_plot, LT_intersect) -> None:
        """Добавляем рисуем модель и торговые уровни в рамках одного трейда."""

        trade = PlotParameters(*trade_plot)

        # Добавление текста заголовка картинки с информацией о торговой паре.
        self.ax.text(0.01, 0.98, trade.ticker,
                     verticalalignment='top', horizontalalignment='left',
                     color='black', fontsize=40, transform=self.ax.transAxes)

        # Рисуем точки модели
        self.add_model_points(model)

        # Рисуем прямые ЛТ и ЛЦ
        CandleStickPlotter.add_model_line(model.lt)
        CandleStickPlotter.add_model_line(model.lc)
        # """ тесты """
        # # Находим бар, который пробил уровень т4
        # first_bar_below_t4 = Point.find_first_bar_by_price(sub_df_for_plot,
        #                                                    model.t4[1],
        #                                                    model.t4[0]+1,
        #                                                    model.properties.dist_cp_t4_x2,
        #                                                    'below'
        #                                                    )
        # # self.ax.plot(first_bar_below_t4[0], first_bar_below_t4[1], marker='v', color='k')
        # if first_bar_below_t4:
        #     t5 = Point.find_t5(sub_df_for_plot, model.t2, model.t4, first_bar_below_t4[0], direction='down_model')
        #     if t5 is not None:
        #         self.ax.plot(t5[0], t5[1], marker='^', color='k')
        #         lt_hp = Line.calculate(model.t3, t5)
        #         if Line.check_line(sub_df_for_plot, lt_hp.slope, lt_hp.intercept, model.t3, t5,
        #                            direction='high'):
        #             # если есть пересечение -
        #             # корректируем линию, добавляя новую точку
        #             t3_1 = Line.correction_LT_down(sub_df_for_plot,
        #                                            model.t3,
        #                                            t5,
        #                                            lt_hp.slope,
        #                                            lt_hp.intercept,
        #                                            return_rightmost=False)
        #             # заново проводим линию ЛТ через т1 и дополнительную точку т3_1
        #             lt_hp = Line.calculate(t3_1, t5)
        #         # Проверяем, что две линии не параллельны друг-другу
        #         if model.lt.slope != lt_hp.slope:
        #             # Находим и определяем желательный угол между линиями
        #             parallel = Line.cos_sim_down(model.lt.slope, lt_hp.slope)
        #             if parallel <= 60:
        #                 CandleStickPlotter.add_model_line(lt_hp)
        #                 # Находим точку пересечения
        #                 hp = Point.find_intersect_two_line_point(lt_hp.intercept,
        #                                                          lt_hp.slope,
        #                                                          model.lc.intercept,
        #                                                          model.lc.slope)
        #                 self.ax.hlines(
        #                     y=t5[1],
        #                     xmin=model.t4[0],
        #                     xmax=model.properties.dist_cp_t4_x2,
        #                     colors='black',
        #                     linestyles='solid',
        #                     linewidth=1,
        #                 )
        #
        #                 self.ax.hlines(
        #                     y=hp[1],
        #                     xmin=model.t4[0],
        #                     xmax=model.properties.dist_cp_t4_x2,
        #                     colors='black',
        #                     linestyles='solid',
        #                     linewidth=1,
        #                 )
        #
        #                 self.ax.hlines(
        #                     y=model.t4[1],
        #                     xmin=0,
        #                     xmax=model.properties.dist_cp_t4_x2,
        #                     colors='black',
        #                     linestyles='solid',
        #                     linewidth=1,
        #                 )
        #
        # """ тесты """
        # Рисуем уровни трейда
        xmin = 0  # начальный индекс дф (если индекс сброшен)
        xmax = len(sub_df_for_plot) - 1  # последний индекс дф
        self.add_trade_lines(model, trade, xmin, xmax)

        if LT_intersect is not None:
            # Рисуем уровень
            plt.vlines(
                ymin=trade.stop_price,
                ymax=trade.entry_price,
                x=LT_intersect[0],
                colors='blue',
                linestyles='solid',
                linewidth=1,
            )

    def show(self) -> None:
        """Показать график."""
        self.ax.show()

    def save(self, filename) -> None:
        """Сохранить график."""
        self.fig.savefig(filename)
