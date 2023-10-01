from core.point_combinations.treand_models.upexpmodel import UpExpModel
from core.point_combinations.treand_models.downexpmodel import DownExpModel
import matplotlib.pyplot as plt
import matplotlib.pyplot as ax
from core.model_utilities.line import Line
from core.model_utilities.point import Point


class TwoModel:
    def __init__(self, up_model: UpExpModel, down_model: DownExpModel):
        self.up_model = up_model
        self.down_model = down_model

    @staticmethod
    def plot_points(ax, points, label: str, color: str):
        x, y = points[0], points[1]
        ax.plot(x, y, marker='o', color=color)
        ax.text(x, y, label, fontsize=10)


    @staticmethod
    def plot_line(ax, line, style: str, color: str, linewidth: float):
        ax.plot(line.points[0], line.points[1], style, color=color,
                linewidth=linewidth)

    @staticmethod
    def plot_model(model, colors):
        TwoModel.plot_points(ax, model.t1, 't1', colors['t1'])
        TwoModel.plot_points(ax, model.t2, 't2', colors['t2'])
        TwoModel.plot_points(ax, model.t3, 't3', colors['t3'])
        TwoModel.plot_points(ax, model.t4, 't4', colors['t4'])

        lc_line = Line.calculate_1(model.CP, model.t4, 0)
        lt_line = Line.calculate_1(model.CP, model.t3, 0)

        TwoModel.plot_line(ax, lc_line, ':', 'purple', 0.9)
        TwoModel.plot_line(ax, lt_line, ':', 'purple', 0.9)

    def find_two_model(self):
        up_models = self.up_model
        down_models = self.down_model
        super_groups = []
        colors_up = {'t1': 'k', 't2': 'k', 't3': 'k', 't4': 'g'}
        colors_down = {'t1': 'r', 't2': 'r', 't3': 'r', 't4': 'b'}
        for up in up_models:
            # up_LT_break_point = Point.find_LT_break_point(up.df,
            #                           up.t4,
            #                           up.properties.dist_cp_t4_x1,
            #                           up.LT.slope,
            #                           up.LT.intercept,
            #                           'up_model'
            #                           )
            # if up_LT_break_point is None:
            #     continue
            # up_LT_break_point_x = int(up_LT_break_point[0])

            for down in down_models:
                # if down.CP[0] > up.t1[0]:
                #     continue
                # Определяем интервал в UpExpModel
                interval_start = up.t4[0]
                # interval_end = up_LT_break_point_x

                # Проверяем, попадают ли точки из DownExpModel в этот интервал
                in_interval_t1 = interval_start <= down.t1[0] <= interval_start + 3
                # in_interval_t3 = interval_start <= down.t3[0] <= interval_end

                # after_interval_t4 = down.t4[0] >= interval_end

                # if down.CP[0] > up.t4[0]:
                #     continue

                if in_interval_t1:
                #     up_dist_t4_lt_break = up.t4[1] - float(up_LT_break_point[1])
                #     up_target_1 = float(up_LT_break_point[1]) - up_dist_t4_lt_break
                #     if up_target_1 >= (
                #     up.df.loc[up_LT_break_point[0]:down.t4[0], 'low'].values
                # ):
                #         continue
                    # dist_t1_t4 = up.t4[0] + (up.t4[0] - up.t1[0])
                    # if down.t1[0] > dist_t1_t4:
                    #     continue
                    # and (down.t1[1] >= up.t4[1])
                    # high_values = df.loc[up.t4[0]:down.t1[0] - 1]['high']
                    #
                    # if not high_values.empty:
                    #     if down.t1[1] != max(high_values):
                    #         continue

                    TwoModel.plot_model(up, colors_up)
                    TwoModel.plot_model(down, colors_down)
                    super_groups.append(TwoModel(up, down))

        return super_groups

    # def find_two_model(self):
    #     up_models = self.up_model
    #     down_models = self.down_model
    #     super_groups = []
    #     colors_up = {'t1': 'k', 't2': 'k', 't3': 'k', 't4': 'g'}
    #     colors_down = {'t1': 'r', 't2': 'r', 't3': 'r', 't4': 'b'}
    #     for down in down_models:
    #         down_LT_break_point = Point.find_LT_break_point(down.df,
    #                                                       down.t4,
    #                                                       down.properties.dist_cp_t4_x2,
    #                                                       down.LT.slope,
    #                                                       down.LT.intercept,
    #                                                       'down_model'
    #                                                         )
    #         if down_LT_break_point is None:
    #             continue
    #         plt.plot(down_LT_break_point[0], down_LT_break_point[1], marker='o', color='r')
    #
    #         down_LT_break_point_x = int(down_LT_break_point[0])
    #         for up in up_models:
    #             # Определяем интервал в UpExpModel
    #             interval_start = down.t4[0]
    #             interval_end = down_LT_break_point_x
    #
    #             # Проверяем, попадают ли точки из DownExpModel в этот интервал
    #             in_interval_t1 = interval_start <= up.t1[0] <= interval_end
    #             in_interval_t3 = interval_start <= up.t3[0] <= interval_end
    #             after_interval_t4 = interval_end < up.t4[0]
    #
    #             if in_interval_t1 and in_interval_t3:
    #                 TwoModel.plot_model(ax, up, colors_up)
    #                 TwoModel.plot_model(ax, down, colors_down)
    #                 super_groups.append(TwoModel(up, down))
    #
    #     return super_groups