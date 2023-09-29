from core.point_combinations.treand_models.up_trend_model import UpTrendModel
from core.model_utilities.distance import Distance
from core.model_utilities.point import Point


class PriceLevelChecker:
    """Проверка достижение уровней активации моделей-кандидатов."""
    def __init__(self, df, candidates, direction='up_model'):
        # self.candidates = UpTrendModel(df).find_candidates()
        self.candidates = candidates
        self.direction = direction
        pass

    def activate_models(self):
        activated_models = []
        not_activated_models = []

        for model in self.candidates:
            dist_cp_t4_x2 = Distance.calculate_x(model.CP, model.t4, 1)
            upper_limit = min(int(dist_cp_t4_x2), len(model.df))

            is_activated, activation_method, variable = self.check_activation(
                model, upper_limit, self.direction)

            if is_activated and activation_method == "first_bar_by_price":
                    t5 = Point.find_t5(model.df, model.t2, model.t4,
                                       variable[0], self.direction)

                    # Если t5 не None, добавляем модель в список активированных
                    if not t5:
                        continue

            model.activation_method = activation_method
            model.activation_variable = variable
            activated_models.append(model)

        return activated_models


    @staticmethod
    def check_activation(model, upper_limit, direction):
        """Проверяет выполнены ли условия активации модели.
        Определяет по какому из двух вариантов произошла активация."""
        # Определяем правую границу поиска


        # Находим бар, который пробил уровень т4
        first_bar_by_price = Point.find_first_bar_by_price(model.df,
                                                           model.t4[1],
                                                           model.t4[0]+1,
                                                           upper_limit,
                                                           direction
                                                           )
        # Находим бар, который пробил ЛТ
        LT_break_point = Point.find_LT_break_point(model.df,
                                                   model.t4,
                                                   upper_limit,
                                                   model.LT.slope,
                                                   model.LT.intercept,
                                                   direction
                                                   )

        if (
                # Если first_bar_above_t4 существует
                # и LT_break_point не существует
                not LT_break_point and first_bar_by_price
                # ИЛИ
                or
                # Если first_bar_above_t4 и LT_break_point оба существуют
                # и индекс first_bar_above_t4 меньше индекса
                # в кортеже LT_break_point
                (LT_break_point and first_bar_by_price
                 and first_bar_by_price[0] < LT_break_point[0])
        ):
            return True, "first_bar_by_price", first_bar_by_price

        elif (LT_break_point and not first_bar_by_price or
              (LT_break_point and first_bar_by_price
               and first_bar_by_price[0] > LT_break_point[0])
        ):
            return True, "LT_break_point", LT_break_point
        else:
            return False, None, None
