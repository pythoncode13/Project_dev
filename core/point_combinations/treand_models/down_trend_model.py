import matplotlib.pyplot as plt

from core.point_combinations.treand_models.trend_model import TrendModel
from core.point_combinations.down_trend_combinations import DownTrendCombinations
from core.model_utilities.line import Line
from core.model_utilities.point import Point
from core.point_combinations.treand_models.downexpmodel import DownExpModel


class DownTrendModel(TrendModel):
    """Модели нисходящего тренда.
    Обращается к классу, отвечающего за поиск точек и их комбинаций.
    Обрабатывает комбинации точек, находя модели.
    Реализован поиск моделей расширения, передача в одноименный класс."""
    def __init__(self, df):
        super().__init__(df)

    def add_combinations(self):
        """Обращается к классу, отвечающего за поиск точек и их комбинаций."""
        return DownTrendCombinations(self.df).add_t4_to_combinations()

    @staticmethod
    def add_lt_line(df, t1, t3, t4):
        """Добавление ЛТ. Проводит прямую, корректирует при необходимости."""
        # проводим линию ЛТ
        LT = Line.calculate(t1, t3)
        # валидация - проверка пересечения "лоу" между двумя точками
        if Line.check_line(df, LT.slope, LT.intercept, t1, t4,
                           direction='high'):
            # если есть пересечение -
            # корректируем линию, добавляя новую точку
            t3_1 = Line.correction_LT_down(df,
                                      t3,
                                      t4,
                                      LT.slope,
                                      LT.intercept,
                                      return_rightmost=True)
            # заново проводим линию ЛТ через т1 и дополнительную точку т3_1
            LT = Line.calculate(t1, t3_1)

        return LT

    @staticmethod
    def add_lc_line(df, t1, t2, t4):
        """Добавление ЛЦ. Проводит прямую, корректирует при необходимости.
        Если коррекция невозможна - возвращает None, что прерывает итерацию."""
        # проводим линию ЛЦ
        LC = Line.calculate(t2, t4)

        # # валидация
        # if Line.check_line(df, LC.slope, LC.intercept, (t1[0] + 1, 0), t4,
        #                    direction='low'):
        #     t4_1 = Line.correction_LC_t4_1_down(df, t2, t4, LC.slope, LC.intercept)
        #     t2_1 = Line.correction_LC_t2_1_down(df, t1, t2, t4_1)
        #
        #     LC = Line.calculate(t2_1, t4_1)
        #     # plt.plot(LC.points[0], LC.points[1], ':',
        #     #          color='purple', linewidth=0.9)
        #     if Line.check_line(df, LC.slope, LC.intercept, (t1[0] + 1, 0),
        #                        t4_1,
        #                        direction='low'):
        #         return None
                # pass
        return LC

    def find_candidates(self):
        """Этот метод объединяет все вышеуказанные методы
        и возвращает найденные точки тренда."""

        combination = self.add_combinations()

        down_model_candidates = []
        for point in combination:
            t1 = point.t1
            t2 = point.t2
            t3 = point.t3
            t4 = point.t4

            # plt.plot(t1[0], t1[1], marker='o', color='k')
            # plt.plot(t4[0], t4[1], marker='o', color='g')
            # Получаем ЛТ, корректируем ее
            LT = self.add_lt_line(self.df, t1, t3, t4)

            # Получаем ЛЦ, корректируем ее
            LC = self.add_lc_line(self.df, t1, t2, t4)
            # Если коррекция невозможна - пропускаем итерацию
            if LC is None:
                continue

            # Проверяем, что две линии не параллельны друг-другу
            if LT.slope == LC.slope:
                continue
            # Находим и определяем желательный угол между линиями
            parallel = Line.cos_sim_down(LT.slope, LC.slope)

            if parallel >= 30:
                continue
            # Находим точку пересечения
            CP = Point.find_intersect_two_line_point(LC.intercept,
                                                     LC.slope,
                                                     LT.intercept,
                                                     LT.slope)
            # Выбираем МР по расположению СТ
            if CP[0] >= t4[0]:
                continue

            down_model_candidates.append(
                DownExpModel(self.df, t1, t2, t3, t4, CP, LT, LC)
            )

        return down_model_candidates
