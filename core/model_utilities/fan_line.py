import matplotlib.pyplot as plt


class FanLine:
    def __init__(self, model):
        # Инициализируем модель
        self.model = model

        self.x_CP, self.y_CP = self.calculate_CP()

    def calculate_CP(self):
        # Точка CP(x, y)
        self.x_CP = self.model.CP[0]
        self.y_CP = self.model.CP[1]
        return self.x_CP, self.y_CP

    def add_lt_line(self):
        # Добавляем линию LT
        y_CP_LT = self.model.LT.slope * self.x_CP + self.model.LT.intercept
        x_LT_extended, y_LT_extended = self.extend_coordinates(self.x_CP, y_CP_LT,
                                                               self.model.LT.points)
        return x_LT_extended, y_LT_extended

    def add_lc_line(self):
        # Добавляем линию LC
        y_CP_LC = self.model.LC.slope * self.x_CP + self.model.LC.intercept
        indices_LC = self.filter_indices()
        x_LC_extended, y_LC_extended = self.extend_coordinates(self.x_CP, y_CP_LC,
                                                               self.model.LC.points,
                                                               indices_LC)
        return x_LC_extended, y_LC_extended

    def filter_indices(self):
        # Фильтруем индексы для LC
        return [i for i, x in enumerate(self.model.LC.points[0]) if
                min(self.model.LT.points[0]) <= x <= max(
                    self.model.LT.points[0])]

    def extend_coordinates(self, x_CP, y_CP_L, points, indices=None):
        # Расширяем координаты, добавляя точку CP
        if indices is None:
            return [x_CP, *points[0]], [y_CP_L, *points[1]]
        else:
            return [x_CP, *[points[0][i] for i in indices]], [y_CP_L,
                                                              *[points[1][i]
                                                                for i in
                                                                indices]]

    def calculate_l0(self):
        # Рассчитываем l0 (это нужно дополнить)
        pass

    def line_plot(self, x_LT, y_LT, x_LC, y_LC):
        # Отрисовываем линии
        plt.plot(x_LT, y_LT, ':', color='purple', linewidth=0.9)
        plt.plot(x_LC, y_LC, ':', color='purple', linewidth=0.9)

    def build_fan(self):
        # Строим весь веер линий
        x_LT, y_LT = self.add_lt_line()
        x_LC, y_LC = self.add_lc_line()
        self.line_plot(x_LT, y_LT, x_LC, y_LC)

        # Здесь можно добавить расчет и отрисовку l0 и других линий

        plt.show()
