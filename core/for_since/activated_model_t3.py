from core.model_utilities.distance import Distance
from core.model_utilities.point import Point
from core.model_utilities.line import Line
from core.for_since.polynom_validator_combination import combinated_validate, length_t1_t3_height_t1_t2_chek
from math import sqrt
import numpy as np
from collections import namedtuple


Combination = namedtuple('Combination', ['t1', 't2', 't3', 'variable', 'LT', 'entry_date', 'up_take_lines'])


def curvature(t1, t2, t3):
    x1, y1 = t1

    x2, y2 = t2

    x3, y3 = t3

    numerator = (x1 - x2) * (y2 - y3) - (y1 - y2) * (x2 - x3)

    denominator = sqrt(((x1 - x3) ** 2 + (y1 - y3) ** 2) * (
                (x1 - x2) ** 2 + (y1 - y2) ** 2) * (
                                   (x2 - x3) ** 2 + (y2 - y3) ** 2))

    if denominator == 0:
        return None  # Return None if the points are collinear, and the curvature is undefined

    return abs(numerator / denominator)


class PriceLevelChecker_t3:
    """Проверка достижение уровней активации моделей-кандидатов."""
    def __init__(self):
        pass

    @staticmethod
    def activate_models(df, three_points_combination):
        activated_models = []
        # not_activated_models = []

        for point in three_points_combination:
            t1 = point[0]
            t2 = point[1]
            t3 = point[2]

            if t3[0] - t2[0] < 2:
                continue
            dist_t1_t3_x5 = Distance.calculate_x(t1, t3, 2)
            upper_limit = min(int(dist_t1_t3_x5), len(df))

            is_activated, activation_method, variable = PriceLevelChecker_t3.check_activation(df,
                point, upper_limit)

            if is_activated and activation_method == 'first_bar_above_t2':
                t4 = (variable, t2[1])
                # LT = UpTrendModel.add_lt_line(df, t1, t3, t4)
                LT = Line.calculate(t1, t3)
                if Line.check_line(df, LT.slope, LT.intercept, t1, t4,
                                   direction='low'):
                    continue

                curvature_value = curvature(t1, t2, t3)
                # if t1[0] == 13:

                t1_1_index = int((t1[0] + t2[0]) / 2)
                t2_1_index = int((t2[0] + t3[0]) / 2)
                # Находим часть DataFrame между индексами t1[0] и t2[0]
                subset_df = df.loc[t1[0]:t2[0]]

                # Находим индекс бара с максимальным значением 'high'
                # t1_1_index = df.loc[t1[0]:t2[0]-1]['high'].idxmax()
                t1_1_price = df.loc[t1_1_index, 'low']
                t1_1 = (t1_1_index, t1_1_price)

                # Находим индекс бара с максимальным значением 'high'
                # t2_1_index = df.loc[t2[0]+1:t3[0]]['high'].idxmax()
                t2_1_price = df.loc[t2_1_index, 'low']
                t2_1 = (t2_1_index, t2_1_price)

                # Соберем их в массивы
                x_values = np.array(
                    [t1[0], t1_1[0], t2[0], t2_1[0], t3[0]])
                y_values = np.array(
                    [t1[1], t1_1[1], t2[1], t2_1[1], t3[1]])

                # Подогнем полином второго порядка и найдем его вторую производную
                coefficients_2 = np.polyfit(x_values, y_values, 2)
                polynomial_2 = np.poly1d(coefficients_2)
                second_derivative_2 = np.polyder(polynomial_2, m=2)
                curvature_values_2 = second_derivative_2(x_values)

                # Подогнем полином третьего порядка и найдем его вторую производную
                coefficients_3 = np.polyfit(x_values, y_values, 3)
                polynomial_3 = np.poly1d(coefficients_3)
                second_derivative_3 = np.polyder(polynomial_3, m=2)
                curvature_values_3 = second_derivative_3(x_values)

                # Фильтрация: проверим, убывают ли значения второй производной
                if all(curvature_values_2[i] < curvature_values_2[i + 1]
                       for i in range(len(curvature_values_2) - 1)):
                    print(
                        "Значения второй производной для полинома 2-го порядка убывают.")
                    continue
                if all(curvature_values_3[i] > curvature_values_3[i + 1]
                       for i in range(len(curvature_values_3) - 1)):
                    print(
                        "Значения второй производной для полинома 3-го порядка убывают.")

                    continue

                print((t1[0], t1[1]), (t2[0], t2[1]), (t3[0], t3[1]))

                subset_df = df.loc[t1[0]:t3[0]]
                # print(subset_df)
                # is_valid = combinated_validate(df, t1, t2, t3)
                # if not is_valid:
                #     continue
                # length_t1_t3, height_t1_t2 = length_t1_t3_height_t1_t2_chek(t1,
                #                                                             t2,
                #                                                             t3)
                # curvature_value = (length_t1_t3 / height_t1_t2)
                up_take_lines = Line.calculate_t3_t2_taget_line(t3, t2)
                entry_date = df.loc[variable, 'dateTime']
                activated_models.append(Combination(t1, t2, t3, variable, LT, entry_date, up_take_lines))

        return activated_models

    @staticmethod
    def check_activation(df, point, upper_limit):
        """Проверяет выполнены ли условия активации модели.
        Определяет по какому из двух вариантов произошла активация."""
        # Определяем правую границу поиска

        # Находим бар, который пробил уровень т4
        first_bar_above_t2 = Point.find_first_bar_by_price(df,
                                                           point[1][1],
                                                           point[2][0] + 1,
                                                           upper_limit,
                                                           direction='above'
                                                           )

        # Находим бар, который пробил ЛТ
        first_bar_below_t3 = Point.find_first_bar_by_price(df,
                                                           point[2][1],
                                                           point[2][0] + 1,
                                                           upper_limit,
                                                           direction='below'
                                                           )

        if (
                # Если first_bar_above_t2 существует
                # и first_bar_below_t3 не существует
                not first_bar_below_t3 and first_bar_above_t2
                # ИЛИ
                or
                # Если first_bar_above_t2 и first_bar_below_t3 оба существуют
                # и индекс first_bar_above_t2 меньше индекса
                # в кортеже first_bar_below_t3
                (first_bar_below_t3 and first_bar_above_t2
                 and first_bar_above_t2 < first_bar_below_t3)
        ):
            return True, "first_bar_above_t2", first_bar_above_t2

        elif (first_bar_below_t3 and not first_bar_above_t2 or
              (first_bar_below_t3 and first_bar_above_t2
               and first_bar_above_t2 > first_bar_below_t3)
        ):
            return True, "first_bar_below_t3", first_bar_below_t3
        else:
            return False, None, None