from core.model_utilities.distance import Distance
from core.model_utilities.line import Line


class UpModelProperty:
    """Класс содержит дополнительные свойства моделей расширения."""
    def __init__(self, df, t1, t2, t3, t4, CP, LT, LC):
        self.df = df
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.CP = CP
        self.LT = LT
        self.LC = LC
        self._dist_cp_t4_x1 = None
        self._dist_cp_t4_x2 = None
        self._CP_to_t1 = None

    """Расстояние по оси x"""
    @property
    def dist_cp_t4_x1(self):
        if self._dist_cp_t4_x1 is None:
            self._dist_cp_t4_x1 = Distance.calculate_x(self.CP, self.t4, 1)
        return self._dist_cp_t4_x1

    @property
    def dist_cp_t4_x2(self):
        if self._dist_cp_t4_x2 is None:
            self._dist_cp_t4_x2 = Distance.calculate_x(self.CP, self.t4, 2)
        return self._dist_cp_t4_x2

    @property
    def CP_to_t1(self):
        self._CP_to_t1 = self.t1[0] - float(self.CP[0])
        return self._CP_to_t1

    @property
    def model_interval(self):
        return self.t4[0] - self.t1[0]

    """Расстояние по оси y"""

    @property
    def up_take_100(self):
        return self.t4[1] + (self.t4[1] - self.t1[1]) * 1

    @property
    def up_take_200(self):
        return self.t4[1] + (self.t4[1] - self.t1[1]) * 2


class ModelPlot:
    def __init__(self, parent_model, start_index):
        self.t1 = (
            int(parent_model.t1[0]) - start_index, parent_model.t1[1])
        self.t2 = (
            int(parent_model.t2[0]) - start_index, parent_model.t2[1])
        self.t3 = (
            int(parent_model.t3[0]) - start_index, parent_model.t3[1])
        self.t4 = (
            int(parent_model.t4[0]) - start_index, parent_model.t4[1])
        self.CP = (
            int(parent_model.CP[0]) - start_index, parent_model.CP[1])
        self.properties = parent_model.properties
        self.lt, self.lc = self.lt_lc_for_plot()

    def lt_lc_for_plot(self):
        """Пересчет координат линий
        для более удобного отображения на графике."""

        # Задаем правую границу прямой
        right_edge_index_lc = (self.t4[0] * 2)
        right_edge_index_lt = (right_edge_index_lc
                               + (self.t4[0] - self.t3[0])
                               )
        # Пересчитываем значения прямых
        lt = Line.calculate_1(self.CP, self.t3, right_edge_index_lt)
        lc = Line.calculate_1(self.CP, self.t4, right_edge_index_lc)

        return lt, lc