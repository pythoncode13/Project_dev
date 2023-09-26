
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class AdvancedModelProperty:
    def __init__(self, model, df):
        self.df = df
        self.t1 = model.t1
        self.t2 = model.t2
        self.t3 = model.t3
        self.t4 = model.t4
        self.CP = model.CP
        self.LT = model.LT
        self.LC = model.LC
        self.dist_cp_t4_x1 = model.properties.dist_cp_t4_x1

    def prepare_data(self):
        """Нормализация данных для использования в других расчетах."""
        # Создаем scaler
        scaler = MinMaxScaler()

        # Выбираем интервал от t1 до t4 и
        # вычисляем среднее между high и low
        interval_data = (
                        self.df.loc[self.t1[0]:self.t4[0], 'high']
                        + self.df.loc[self.t1[0]:self.t4[0], 'low']
                        ) / 2

        # Нормализуем данные
        y_data_normalized = scaler.fit_transform(
            np.array(interval_data).reshape(-1, 1))

        return y_data_normalized.ravel()

    @staticmethod
    def compute_mse(df, normalized_data, start, end):
        # Вычисляем количество элементов в интервале
        num_elements = len(df.loc[start[0]:end[0]])

        # Выбираем соответствующие элементы из нормализованного массива
        normalized_interval_data = normalized_data[:num_elements]

        # Ваши данные
        x_data = np.array(range(len(normalized_interval_data)))

        # Подгоняем данные под параболу
        coeffs = np.polyfit(x_data, normalized_interval_data, deg=2)

        # Вычисляем предсказанные значения
        y_fit = np.polyval(coeffs, x_data)

        # Вычисляем среднеквадратичную ошибку
        mse = np.mean((normalized_interval_data - y_fit) ** 2)

        return mse

    @property
    def get_mse(self):

        normalized_data = self.prepare_data()

        mse_t1_t2 = self.compute_mse(self.df,
                                     normalized_data,
                                     self.t1,
                                     self.t2)

        mse_t2_t3 = self.compute_mse(self.df,
                                     normalized_data,
                                     self.t2,
                                     self.t3)

        mse_t3_t4 = self.compute_mse(self.df,
                                     normalized_data,
                                     self.t3,
                                     self.t4)

        mse_t1_t3 = self.compute_mse(self.df,
                                     normalized_data,
                                     self.t1,
                                     self.t3)

        mse = mse_t1_t2 + mse_t2_t3

        return mse_t1_t2, mse_t2_t3, mse_t3_t4, mse_t1_t3, mse

    @property
    def calculate_golden_ratio(self):
        # Рассчитаем отношения
        ratio1 = abs((self.t2[0] - self.t1[0]) / (self.t3[0] - self.t1[0]))
        ratio2 = abs((self.t4[0] - self.t3[0]) / (self.t4[0] - self.t1[0]))

        # Золотое сечение
        golden_ratio = (1 + 5 ** 0.5) / 2  # равно примерно 1.618

        # Сравним наши отношения со золотым сечением
        ratio1_golden = ratio1 - golden_ratio
        ratio2_golden = ratio2 - golden_ratio
        ratio1_ratio2 = ratio1_golden / ratio2_golden
        ratio2_ratio1 = ratio2_golden / ratio1_golden

        return ratio1_golden, ratio2_golden, ratio1_ratio2, ratio2_ratio1

    # def calc_num_of_distances_low_to_t1up(self):
    #     df_valid = (
    #         self.df.loc[:self.t1[0]]
    #         [self.df.loc[:self.t1[0]]['low'] < self.t1[1]]
    #     )
    #
    #     first_bar_index_before_t1up = df_valid.last_valid_index()
    #
    #     if first_bar_index_before_t1up is not None:
    #
    #         num_of_distances_low_to_t1up = (
    #                 (self.t1[0] - first_bar_index_before_t1up)
    #                 / self.dist_cp_t4_x1
    #         )
    #
    #     else:
    #         num_of_distances_low_to_t1up = 1000
    #
    #     return num_of_distances_low_to_t1up
    #
    # def calc_num_of_distances_high_to_t4up(self):
    #     df_valid = (
    #         self.df.loc[:self.t4[0]]
    #         [self.df.loc[:self.t4[0]]['high'] > self.t4[1]]
    #     )
    #
    #     first_bar_index_before_t4up = df_valid.last_valid_index()
    #
    #     if first_bar_index_before_t4up is not None:
    #
    #         num_of_distances_high_to_t4up = (
    #                 (self.t4[0] - first_bar_index_before_t4up)
    #                 / self.dist_cp_t4_x1
    #         )
    #
    #     else:
    #         num_of_distances_high_to_t4up = 1000
    #
    #     return num_of_distances_high_to_t4up
    #
    # @property
    # def nums_of_distances_to_points(self):
    #
    #     num_of_distances_low_to_t1up = self.calc_num_of_distances_low_to_t1up()
    #     num_of_distances_high_to_t4up = self.calc_num_of_distances_high_to_t4up()
    #
    #     return num_of_distances_low_to_t1up, num_of_distances_high_to_t4up

    def calc_num_of_distances_to_t1(self, direction):

        if direction == 'up_model':
            condition = self.df.loc[:self.t1[0]]['low'] < self.t1[1]
        else:
            condition = self.df.loc[:self.t1[0]]['high'] > self.t1[1]

        df_valid = self.df.loc[:self.t1[0]][condition]
        first_bar_index_before_t1 = df_valid.last_valid_index()

        if first_bar_index_before_t1 is not None:
            num_of_distances_to_t1 = (
                    (self.t1[0] - first_bar_index_before_t1)
                    / self.dist_cp_t4_x1
            )
        else:
            num_of_distances_to_t1 = 1

        return num_of_distances_to_t1

    def calc_num_of_distances_to_t4(self, direction):

        if direction == 'up_model':
            condition = self.df.loc[:self.t4[0]]['high'] > self.t4[1]
        else:
            condition = self.df.loc[:self.t4[0]]['low'] < self.t4[1]

        df_valid = self.df.loc[:self.t4[0]][condition]
        first_bar_index_before_t4 = df_valid.last_valid_index()

        if first_bar_index_before_t4 is not None:
            num_of_distances_to_t4 = (
                    (self.t4[0] - first_bar_index_before_t4)
                    / self.dist_cp_t4_x1
            )
        else:
            num_of_distances_to_t4 = 1

        return num_of_distances_to_t4

    def nums_of_distances_to_points(self, direction):
        num_of_distances_to_t1 = self.calc_num_of_distances_to_t1(direction)
        num_of_distances_to_t4 = self.calc_num_of_distances_to_t4(direction)

        return (
            num_of_distances_to_t1,
            num_of_distances_to_t4
        )
