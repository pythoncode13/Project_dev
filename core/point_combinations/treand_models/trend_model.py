class TrendModel:
    def __init__(self, df):
        self.df = df
        self.trend_points = []
        self.combinations = self.add_combinations()

    def add_combinations(self):
        # Общая логика или пустой метод
        pass

    # Остальной код...



# class DownTrendModel(TrendModel):
#     def add_combinations(self):
#         return add_t4_to_combinations_down(self.df, range_limit=30)  # Функция для даун модели
#
#     # Остальной код...
