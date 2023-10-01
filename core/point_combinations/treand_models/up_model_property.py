from core.model_utilities.distance import Distance
from core.model_utilities.line import Line
from core.model_utilities.point import Point


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
        self._LT_break_point = None
        self._target_1 = None
        self._target_3 = None
        self._target_5 = None

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

    @property
    def LT_break_point(self):
        self._LT_break_point = Point.find_LT_break_point(self.df,
                                                      self.t4,
                                                      self.dist_cp_t4_x2,
                                                      self.LT.slope,
                                                      self.LT.intercept,
                                                      'up_model'
                                                      )
        return self._LT_break_point

    @property
    def target_1(self):
        if self.LT_break_point:
            max_price = self.df['high'].iloc[
                        int(self.t4[0]):int(self.LT_break_point[0])
                        ].max()
            dist_t4_lt_break = max_price - float(self.LT_break_point[1])
            self._target_1 = float(self.LT_break_point[1]) - dist_t4_lt_break
        return self._target_1

    @property
    def target_3(self):
        if self.LT_break_point:
            high_t4_t1 = self.t4[1] - self.t1[1]
            # self._target_3 = float(self.LT_break_point[1]) - high_t4_t1
            self._target_3 = self.t1[1] - high_t4_t1

        return self._target_3

    @property
    def target_5(self):
        if self.LT_break_point:
            max_price = self.df['high'].iloc[
                        int(self.t4[0]):int(self.LT_break_point[0])
                        ].max()
            print(max_price)
            dist_t4_lt_break = max_price - float(self.CP[1])
            self._target_5 = float(self.CP[1]) - dist_t4_lt_break
        return self._target_5


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

        self.parent_model_LT_break_point = parent_model.properties.LT_break_point
        self.start_index = start_index
        self._LT_break_point = None

        self.lt, self.lc = Line.lt_lc_for_plot(self.CP, self.t3, self.t4)

    @property
    def LT_break_point(self):
        if self.parent_model_LT_break_point:
            self._LT_break_point = (
                self.parent_model_LT_break_point[0] - self.start_index,
                self.parent_model_LT_break_point[1])
        return self._LT_break_point
