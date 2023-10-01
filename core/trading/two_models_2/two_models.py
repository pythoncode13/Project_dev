from core.point_combinations.treand_models.upexpmodel import UpExpModel
from core.point_combinations.treand_models.downexpmodel import DownExpModel
import matplotlib.pyplot as plt
import matplotlib.pyplot as ax
from core.model_utilities.line import Line
from core.model_utilities.point import Point

from decimal import Decimal, getcontext

getcontext().prec = 10


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
            for down in down_models:
                # if down.CP[0] > up.t1[0]:
                #     continue
                # Определяем интервал в UpExpModel
                interval_start = up.t4[0]-1
                interval_end = up.t4[0]+1

                # Проверяем, попадают ли точки из DownExpModel в этот интервал
                # Точка т1 довн == т4 ап
                in_interval_t1 = interval_start <= down.t1[0] <= interval_end

                if not in_interval_t1:
                    continue

                # т3 над ЛТ_ап
                # Вычисляем значение на прямой up.LT в точке x_t3
                y_LT = up.LT.slope * Decimal(down.t3[0]) + up.LT.intercept
                print(y_LT)

                # Проверяем, находится ли точка выше прямой
                if down.t3[1] < y_LT:
                    continue

                # т4 под ЛТ_ап
                # Вычисляем значение на прямой up.LT в точке x_t3
                y_LT = up.LT.slope * Decimal(down.t4[0]) + up.LT.intercept

                # Проверяем, находится ли точка выше прямой
                if down.t4[1] > y_LT:
                    continue

                # т2 над ЛТ_ап
                # Вычисляем значение на прямой up.LT в точке x_t2
                y_LT = up.LT.slope * Decimal(down.t2[0]) + up.LT.intercept
                print(y_LT)

                # Проверяем, находится ли точка выше прямой
                if down.t2[1] < y_LT:
                    continue

                # if not up.properties.target_1:
                #     continue
                # if down.t4[1] > up.properties.target_1:
                #     continue
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

                # TwoModel.plot_model(up, colors_up)
                # TwoModel.plot_model(down, colors_down)
                super_groups.append(TwoModel(up, down))

        return super_groups
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

# in_interval_t3 = interval_start <= down.t3[0] <= interval_end

                # after_interval_t4 = down.t4[0] >= interval_end

                # if down.CP[0] > up.t4[0]:
                #     continue