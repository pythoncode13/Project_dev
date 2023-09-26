import numpy as np
from core.models.up_model_property import Line, Point


class UpModel:
    def __init__(
            self,
            # df,
            t1up,
            t2up,
            t3up,
            t4up,
            t5up,
            first_bar_above_t4up,
            up_take_lines,
            dist_cp_t4_x2,
            HP_up_point,
            LC_break_point,
            point_under_LC,
            LT_break_point,
            CP_up_point,
            dist_cp_t4_x1,
            LT_break_point_close,
            slope_LT,
            intercept_LT,
            # LC_up,
            # slope_LC,
            # intercept_LC,
            # LT_up,
            # slope_LT,
            # intercept_LT,
            # CP_up_point
    ):
        # self.df = df
        self.t1up = t1up
        self.t2up = t2up
        self.t3up = t3up
        self.t4up = t4up
        self.t5up = t5up
        self.first_bar_above_t4up = first_bar_above_t4up
        self.up_take_lines = up_take_lines
        self.dist_cp_t4_x2 = dist_cp_t4_x2
        self.HP_up_point = HP_up_point
        self.LC_break_point = LC_break_point
        self.point_under_LC = point_under_LC
        self.LT_break_point = LT_break_point
        self.CP_up_point = CP_up_point
        self.dist_cp_t4_x1 = dist_cp_t4_x1
        self.LT_break_point_close = LT_break_point_close
        self.slope_LT = slope_LT
        self.intercept_LT = intercept_LT
        # self.LC_up = LC_up
        # self.slope_LC = slope_LC
        # self.intersect_LC = intercept_LC
        # self.LT_up = LT_up
        # self.slope_LT = slope_LT
        # self.intersect_LT = intercept_LT
        # self.CP_up_point = t1up
        # self.dist_cp_t4_x1 = None
        # self.dist_cp_t4_x2 = None
        # self.find_LT_up_breakout_point()
        # self.find_target()
        # self.find_first_bar_above_t4up()
        # self.above_is_faster_breakout()
        # self.find_t5up()


    '''
    # # Создаем словарь для хранения объектов UpModel с уникальными t1up
        # unique_t1up_models = {}
        #
        # # Проходим по списку объектов
        # for model in up_trend_points:
        #     # Округляем и приводим к целому значение t4up[0] для текущей модели
        #     t4up_current_rounded = int(np.round(model.t4up[0]))
        #
        #     # Проверяем, не попадает ли t4up[0] данной модели в диапазон t4up[0] - 7 : t4up[0] других моделей
        #     if any(int(np.round(other_model.t4up[0])) in range(
        #             t4up_current_rounded - 7, t4up_current_rounded + 1) for
        #            other_model in up_trend_points if model != other_model):
        #         # Если условие выполняется, пропускаем данную модель и переходим к следующей
        #         continue
        #
        #     # Извлекаем t1up из текущего объекта
        #     t1up_current = tuple(model.t1up)
        #
        #     # Если текущий t1up уже есть в словаре
        #     if t1up_current in unique_t1up_models:
        #         # Если t4up[0] текущего объекта меньше t4up[0] сохраненного
        #         if model.t4up[0] < unique_t1up_models[t1up_current].t4up[0]:
        #             # Заменяем сохраненный объект на текущий
        #             unique_t1up_models[t1up_current] = model
        #     else:
        #         # Если текущий t1up отсутствует в словаре, добавляем его
        #         unique_t1up_models[t1up_current] = model
        #
        # # Преобразуем значения словаря обратно в список
        # up_trend_points = list(unique_t1up_models.values())
        '''

    @staticmethod
    def unique_models(up_trend_points):
        """Фильтрует модели, оставляя модели нулевого прохода."""
        # Создаем словарь для хранения объектов UpModel с уникальными t1up
        unique_t1up_models = {}

        # Проходим по списку объектов
        for model in up_trend_points:
            # Округляем и приводим к целому значение t4up[0] для текущей модели
            t4up_current_rounded = int(round(model.t4up[0]))

            # Проверяем, не попадает ли t4up[0] данной модели в диапазон t4up[0] - 7 : t4up[0] других моделей
            if not any(t4up_current_rounded - 7 <= int(
                    round(other_model.t4up[0])) <= t4up_current_rounded
                       for other_model in up_trend_points
                       if model != other_model):
                # Извлекаем t1up из текущего объекта
                t1up_current = tuple(model.t1up)

                # Заменяем сохраненный объект на текущий, если t4up[0] текущего объекта меньше t4up[0] сохраненного
                # Или добавляем его, если текущий t1up отсутствует в словаре
                if t1up_current not in unique_t1up_models or model.t4up[0] < \
                        unique_t1up_models[t1up_current].t4up[0]:
                    unique_t1up_models[t1up_current] = model

        # Преобразуем значения словаря обратно в список
        return list(unique_t1up_models.values())

    @property
    def find_dist_cp_t4_x1(self):
        if self.dist_cp_t4_x1 is None:
            self.dist_cp_t4_x1 = self.t4up[0] + (self.t4up[0] - self.CP_up_point[0])
        return self.dist_cp_t4_x1

    @property
    def find_dist_cp_t4_x2(self):
        if self.dist_cp_t4_x2 is None:
            self.dist_cp_t4_x2 = self.t4up[0] + ((
                    self.t4up[0] - self.CP_up_point[0]) * 2)
        return self.dist_cp_t4_x2

    def find_LT_up_breakout_point(self):
        index_array = np.arange(self.t4up[0] + 1, len(self.df))
        expected_price_array = self.slope_LT * index_array + self.intersect_LT

        # получаем булев массив, где True - точка пересечения
        intersects = expected_price_array >= self.df['low'][index_array].values
        if intersects.any():
            intersection_index = index_array[intersects][0]
            self.LT_up_breakout_point = (
            intersection_index, expected_price_array[intersects][0])
            return self.LT_up_breakout_point
        return None

    def find_first_bar_above_t4up(self):
        """
        Ищет первый бар, хай которого выше уровня t4up[1] после t4up[0]+1
        """
        for i in range(self.t4up[0], len(self.df)):
            if self.df.loc[i, 'high'] > self.t4up[1]:
                self.first_bar_above_t4up = i
                return self.first_bar_above_t4up  # Возвращаем индекс первого бара, удовлетворяющего условию
        return None  # Возвращаем None, если такого бара не найдено

    def above_is_faster_breakout(self):
        self.LT_up_breakout_point = self.find_LT_up_breakout_point()
        self.first_bar_above_t4up = self.find_first_bar_above_t4up()

        if self.first_bar_above_t4up and not self.LT_up_breakout_point:
            return True
        if self.first_bar_above_t4up and self.LT_up_breakout_point and self.first_bar_above_t4up < self.LT_up_breakout_point[0]:
            return True
        return False

    def find_max_high_between_t4up_and_lt_up(self):
        max_high = None
        max_high_index = None
        passed_t4up = False
        high_below_LT_up = False

        for index, row in self.df.iterrows():
            if not passed_t4up:
                if index == self.t4up[0]:
                    passed_t4up = True
                    max_high = row['high']
                    max_high_index = index
                continue

            high = row['high']
            close = row['close']

            if close < self.LT_up_breakout_point[1]:
                break

            if high > max_high:
                max_high = high
                max_high_index = index
        return max_high_index, max_high

    def find_target(self):
        LT_breakout_point = self.find_LT_up_breakout_point()
        if LT_breakout_point is not None:
            max_high_index, max_high = self.find_max_high_between_t4up_and_lt_up()
            self.up_tg1 = LT_breakout_point[1] - (max_high - LT_breakout_point[1]) * 1.0
            self.up_tg2 = self.t1up[1]
            self.up_tg3 = LT_breakout_point[1] - (self.t4up[1] - self.t1up[1]) * 1.0
            self.up_tg5 = self.CP_up_point[1] - (max_high - self.CP_up_point[1])
        else:
            self.up_tg1 = self.up_tg2 = self.up_tg3 = self.up_tg5 = None
        return self.up_tg1, self.up_tg2, self.up_tg3, self.up_tg5

    def find_t5up(self):
        if self.above_is_faster_breakout():
            first_bar_above_t4up = self.find_first_bar_above_t4up()

            t5up_index = self.df.loc[self.t4up[0] + 1:first_bar_above_t4up,
                         'low'].idxmin()
            t5up_price = self.df.loc[t5up_index, 'low']
            t5up = (t5up_index, t5up_price)
            # Находим индексы баров между t3 и t5
            lows = self.df.loc[self.t3up[0]: t5up[0], 'low']

            # Вычисляем slope и intercept для линии от t3up до t5up
            slope_t3_t5, intercept_t3_t5, _ = Line.calculate(self.t3up, t5up)

            # Находим бары, лоу которых пересекает прямую
            intersects = lows < (slope_t3_t5 * lows.index + intercept_t3_t5)

            if not intersects.any():
                # Если пересечений нет, возвращаем None
                return None, None, None

            # Выбираем индексы пересечений и переворачиваем их
            intersect_indices = lows[intersects].index[::-1]

            # Инициализируем t3up1 как None до цикла for
            t3up1 = None

            # Итерируемся по пересекающим индексам справа налево
            for intersect_index in intersect_indices:
                # Вычисляем потенциальные slope и intercept
                slope_candidate, intercept_candidate, _ = Line.calculate(
                    (intersect_index, lows.loc[intersect_index]), t5up)

                # Если slope_candidate является None, пропускаем текущую итерацию
                if slope_candidate is None:
                    continue

                # Проверяем, есть ли бары, которые пересекают новую прямую
                lows_between = self.df.loc[self.t3up[0]: t5up[0], 'low']
                intersects_between = lows_between < (
                            slope_candidate * lows_between.index + intercept_candidate)

                # Если таких баров нет, принимаем текущий slope и intercept и определяем t3up1
                if not intersects_between.any():
                    t3up1 = (intersect_index, lows.loc[intersect_index])
                    break
            # Здесь вы должны проверить, был ли t3up1 изменен.
            # Если он все еще None, это значит, что цикл не нашел подходящего значения, и вам нужно принять соответствующие меры.
            if t3up1 is None:
                slope, intercept, LC_HP = Line.calculate(self.t3up, t5up)

                if slope is not None:
                    if slope != self.slope_LC and Line.cosine_similarity(slope, self.slope_LC) < 0.001:
                        x_intersect_LC_LT_point = (
                                                          self.intersect_LC - intercept) / (
                                                          slope - self.slope_LC)
                        y_intersect_LC_LT_point = slope * x_intersect_LC_LT_point + intercept

                        if x_intersect_LC_LT_point >= self.t4up[0]:
                            HP_up_point = (
                                x_intersect_LC_LT_point,
                                y_intersect_LC_LT_point)
                        else:
                            HP_up_point = None
                    else:
                        HP_up_point = None
                else:
                    HP_up_point = None

                return t5up, HP_up_point, LC_HP

            slope, intercept, LC_HP = Line.calculate(t3up1, t5up)

            if slope is not None:
                if slope != self.slope_LC:
                    x_intersect_LC_LT_point = (
                                                          self.intersect_LC - intercept) / (
                                                          slope - self.slope_LC)
                    y_intersect_LC_LT_point = slope * x_intersect_LC_LT_point + intercept

                    if x_intersect_LC_LT_point >= self.t4up[0]:
                        HP_up_point = (
                        x_intersect_LC_LT_point, y_intersect_LC_LT_point)
                    else:
                        HP_up_point = None
                else:
                    HP_up_point = None
            else:
                HP_up_point = None

            return t5up, HP_up_point, LC_HP

            # # Вычисляем пересечения
            # intersects = lows < (slope * lows.index + intercept)
            #
            # # Если есть пересечения
            # if intersects.any():
            #     # Выбираем индексы пересечений
            #     intersect_indices = lows[intersects].index
            #
            #     diff_t5up_intercept = intersect_indices - t5up[0]
            #     angle_denominator = t5up[1] - lows.loc[intersect_indices]
            #
            #     for intersect_index, angle_denom, diff in zip(intersect_indices,
            #                                                   angle_denominator,
            #                                                   diff_t5up_intercept):
            #         # Вычисляем новые углы и коэффициенты
            #         angle = np.arctan2(angle_denom, diff)
            #         slope_candidate = np.tan(angle)
            #         intercept_candidate = lows.loc[
            #                                   intersect_index] - slope_candidate * intersect_index
            #
            #         # Проверяем, есть ли свечи слева, которые пересекают прямую
            #         lows_left = self.df.loc[self.t3up[0]: intersect_index - 1, 'low']
            #         intersects_left = lows_left < (
            #                 slope_candidate * lows_left.index + intercept_candidate)
            #
            #         # Если есть бары слева, которые пересекают линию, пропускаем текущий индекс пересечения
            #         if intersects_left.any():
            #             continue
            #
            #         # Если нет баров слева, которые пересекают линию, принимаем текущие slope и intercept
            #         slope = slope_candidate
            #         intercept = intercept_candidate
            #         break
            #
            # # # Получаем переменную прямой
            # # x = np.linspace(self.t3up[0], t5up[0] + ((t5up[0] - self.t3up[0]) * 3), 300)
            # # y = slope * x + intercept
            # # LT_HP = (x, y)
            #
            # if slope != self.slope_LC:
            #     x_intersect_LC_LT_point = (self.intersect_LC - intercept) / (
            #             slope - self.slope_LC)
            #     y_intersect_LC_LT_point = slope * x_intersect_LC_LT_point + intercept
            #
            #     if x_intersect_LC_LT_point >= self.t4up[0]:
            #         HP_up_point = (
            #             x_intersect_LC_LT_point, y_intersect_LC_LT_point)
            #     else:
            #         HP_up_point = None
            # else:
            #     HP_up_point = None
            #
            # # Возвращаем точку, которая продлевает линию, и прямую
        #     return t5up
        # else:
        #     None