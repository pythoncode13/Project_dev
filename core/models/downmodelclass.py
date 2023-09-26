class DownModel:
    def __init__(
            self,
            # df,
            t1down,
            t2down,
            t3down,
            t4down,
            t5down,
            first_bar_above_t4down,
            down_take_lines,
            dist_cp_t4_x2,
            HP_down_point,
            LC_break_point,
            point_under_LC,
            LT_break_point,
            CP_down_point,
            dist_cp_t4_x1,
            LT_break_point_close,
            slope_LT,
            intercept_LT,

    ):
        # self.df = df
        self.t1down = t1down
        self.t2down = t2down
        self.t3down = t3down
        self.t4down = t4down
        self.t5down = t5down
        self.first_bar_above_t4down = first_bar_above_t4down
        self.down_take_lines = down_take_lines
        self.dist_cp_t4_x2 = dist_cp_t4_x2
        self.HP_down_point = HP_down_point
        self.LC_break_point = LC_break_point
        self.point_under_LC = point_under_LC
        self.LT_break_point = LT_break_point
        self.CP_down_point = CP_down_point
        self.dist_cp_t4_x1 = dist_cp_t4_x1
        self.LT_break_point_close = LT_break_point_close
        self.slope_LT = slope_LT
        self.intercept_LT = intercept_LT

    @staticmethod
    def unique_models(down_trend_points):
        """Фильтрует модели, оставляя модели нулевого прохода."""
        # Создаем словарь для хранения объектов DownModel с уникальными t1up
        unique_t1down_models = {}

        # Проходим по списку объектов
        for model in down_trend_points:
            # Округляем и приводим к целому значение t4up[0] для текущей модели
            t4down_current_rounded = int(round(model.t4down[0]))

            # Проверяем, не попадает ли t4down[0] данной модели в диапазон t4up[0] - 7 : t4down[0] других моделей
            if not any(t4down_current_rounded - 7 <= int(round(other_model.t4down[0])) <= t4down_current_rounded
                       for other_model in down_trend_points
                       if model != other_model):
                # Извлекаем t1down из текущего объекта
                t1down_current = tuple(model.t1down)

                # Заменяем сохраненный объект на текущий, если t4down[0] текущего объекта больше t4down[0] сохраненного
                # Или добавляем его, если текущий t1down отсутствует в словаре
                if t1down_current not in unique_t1down_models or model.t4down[0] > unique_t1down_models[t1down_current].t4down[0]:
                    unique_t1down_models[t1down_current] = model

        # Преобразуем значения словаря обратно в список
        return list(unique_t1down_models.values())
