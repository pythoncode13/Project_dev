from core.point_combinations.treand_models.trend_model import TrendModel
from core.point_combinations.up_trend_combinations import UpTrendCombinations
from core.model_utilities.line import Line
from core.model_utilities.point import Point
from core.model_utilities.distance import Distance
from core.point_combinations.treand_models.upexpmodel import UpExpModel


class UpTrendModel(TrendModel):
    """Модели восходящего тренда.
    Обращается к классу, отвечающего за поиск точек и их комбинаций.
    Обрабатывает комбинации точек, находя модели.
    Реализован поиск моделей расширения, передача в одноименный класс."""
    def __init__(self, df):
        super().__init__(df)

    def add_combinations(self):
        """Обращается к классу, отвечающего за поиск точек и их комбинаций."""
        return UpTrendCombinations(self.df).add_t4_to_combinations()

    @staticmethod
    def add_lt_line(df, t1, t3, t4):
        """Добавление ЛТ. Проводит прямую, корректирует при необходимости."""
        # проводим линию ЛТ
        LT = Line.calculate(t1, t3)
        # валидация - проверка пересечения "лоу" между двумя точками
        if Line.check_line(df, LT.slope, LT.intercept, t1, t4,
                           direction='low'):
            # если есть пересечение -
            # корректируем линию, добавляя новую точку
            t3_1 = Line.correction_LT(df,
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
        t2_01 = t2
        # валидация
        # if Line.check_line(df, LC.slope, LC.intercept, (t1[0] + 1, 0), t4,
        #                    direction='close'):
        #     return None, None
        # if Line.check_line(df, LC.slope, LC.intercept, (t1[0] + 1, 0), t4,
        #                    direction='high'):
        #     t4_1 = Line.correction_LC_t4up1(df, t2, t4, LC.slope, LC.intercept)
        #     t2_1 = Line.correction_LC_t2up1(df, t1, t2, t4_1)
        #
        #     LC = Line.calculate(t2_1, t4_1)
        #     t2_01 = t2_1
        #     if Line.check_line(df, LC.slope, LC.intercept, (t1[0] + 1, 0),
        #                        t4_1,
        #                        direction='high'):
        #         return None, None

        return LC, t2_01

    def find_candidates(self):
        """Этот метод объединяет все вышеуказанные методы
        и возвращает найденные точки тренда."""

        combination = self.add_combinations()

        up_model_candidates = []
        for point in combination:
            t1 = point.t1
            t2 = point.t2
            t3 = point.t3
            t4 = point.t4

            # Получаем ЛТ, корректируем ее
            LT = self.add_lt_line(self.df, t1, t3, t4)
            # Получаем ЛЦ, корректируем ее
            LC, t2_1 = self.add_lc_line(self.df, t1, t2, t4)
            # Если коррекция невозможна - пропускаем итерацию
            if LC is None:
                continue
            # Проверяем, что две линии не параллельны друг-другу
            if LT.slope == LC.slope:
                continue
            # Находим и определяем желательный угол между линиями
            parallel = Line.cos_sim(LT.slope, LC.slope)
            #TODO было 30
            if parallel >= 60:
                continue
            # Находим точку пересечения
            CP = Point.find_intersect_two_line_point(LC.intercept,
                                                     LC.slope,
                                                     LT.intercept,
                                                     LT.slope)
            # Выбираем МР по расположению СТ
            if CP[0] >= t4[0]:
                continue
            # Добавления:
            CP_n, t1_n, t2_n, t3_n, t4_n = Point.normalize_points([CP, t1, t2, t3, t4])
            # Сила модели
            if (int(t3[0]) - int(t1[0])) < (int(t1[0]) - int(CP[0])):
                continue
            dist_CP_t1_n = Distance.calculate_distance(CP_n, t1_n)
            dist_t1_t3_n = Distance.calculate_distance(t1_n, t3_n)
            # if dist_CP_t1_n > dist_t1_t3_n:
            #     continue
            # т4 в два раза выше т2
            dist_h_t1_t2_n = t2_n[1] - t1_n[1]
            dist_h_t2_t4_n = t4_n[1] - t2_n[1]
            # print(self.df.loc[t1[0]])
            # print('dist_h_t1_t2_n :', dist_h_t1_t2_n)
            # print('dist_h_t2_t4_n :', dist_h_t2_t4_n)
            # if dist_h_t1_t2_n * 0.8 > dist_h_t2_t4_n:
            #     continue
            up_model_candidates.append(
                UpExpModel(self.df, t1, t2, t3, t4, CP, LT, LC, t2_1, parallel)
            )

        return up_model_candidates
