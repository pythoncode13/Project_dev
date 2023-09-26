from core.point_combinations.treand_models.up_model_property import UpModelProperty


class UpExpModel:
    """Класс содержит модели расширения."""
    def __init__(self, df, t1, t2, t3, t4, CP, LT, LC, t2_1, parallel, activation_method=None, activation_variable=None):
        self.df = df
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.CP = CP
        self.LT = LT
        self.LC = LC
        self.t2_1 = t2_1
        self.parallel = parallel

        self.activation_method = activation_method
        self.activation_variable = activation_variable
        self.properties = UpModelProperty(df, t1, t2, t3, t4, CP, LT, LC)
