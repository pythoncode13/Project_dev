import numpy as np
from scipy.stats import skew, kurtosis, kstest
from math import atan2, degrees, sqrt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from datetime import datetime


class NewParams:
    def __init__(self, df, model):
        self.df = df
        self.model = model
        self.normalized_data = self.get_normalized_data()
        self.mse_d1_t1_t4, self.x_data_d1_t1_t4 = self.get_mse_d1_t1_t4()
        self.interval_data, self.y_predicted = self.model_ridge_regression()
        self.coeffs_d4_t1_t4_orig = self.get_mse_d4_t1_t4()
        self.center_of_mass = self.calc_center_of_mass_without_weight()
        self.center_of_mass_with_CP = self.calc_center_of_mass_with_CP()

    @staticmethod
    def calculate_statistics(df, columns, start_index, end_index):
        # Форма области между прямыми
        values = df.loc[start_index:end_index, columns].values.flatten()
        return values.mean(), values.std(), skew(values)

    @staticmethod
    def compute_mse_for_degree(df, normalized_data, start, end, degree):
        num_elements = len(df.loc[start[0]:end[0]])
        normalized_interval_data = normalized_data[:num_elements]
        x_data = np.array(range(len(normalized_interval_data)))
        coeffs = np.polyfit(x_data, normalized_interval_data, deg=degree)
        y_fit = np.polyval(coeffs, x_data)
        mse = mean_squared_error(normalized_interval_data, y_fit)
        return mse, coeffs, x_data

    @staticmethod
    def prepare_data(df, t1up, t4up):
        scaler = MinMaxScaler()
        interval_data = (df.loc[t1up[0]:t4up[0], 'high'] + df.loc[
                                                           t1up[0]:t4up[0],
                                                           'low']) / 2
        y_data_normalized = scaler.fit_transform(
            np.array(interval_data).reshape(-1, 1))
        return y_data_normalized.ravel()

    @staticmethod
    def compute_skewness_kurtosis(y_real, y_predicted):
        # Функция для вычисления асимметрии и эксцесса
        residuals = y_real - y_predicted
        skewness = skew(residuals)
        kurt = kurtosis(residuals)
        return skewness, kurt

    def statistic_params_beetwen_lines(self):
        "Среднее: mean"
        "Стандартное отклонение: std_dev"
        "Асимметрия: skewness"
        # Форма области между прямыми

        mean_open, std_dev_open, skewness_open = NewParams.calculate_statistics(
            self.df, ['open'], self.model.t1[0], self.model.t4[0])

        return skewness_open

    @staticmethod
    # Функция для преобразования строки даты и времени в числовой формат (timestamp)
    def to_timestamp(date_str):
        # Формат даты и времени
        date_format = "%Y-%m-%d %H-%M-%S"
        # Преобразование строки в timestamp, если date_str является строкой, иначе возврат date_str
        return datetime.strptime(date_str,
                                 date_format).timestamp() if isinstance(
            date_str, str) else date_str

    @staticmethod
    def compute_derivative(coeffs, x_point):
        # Функция для вычисления производной полинома в заданной точке
        # Вычисление коэффициентов производной полинома
        derivative_coeffs = np.polyder(coeffs)
        # Вычисление значения производной в точке x_point (предварительно преобразованной в timestamp)
        return np.polyval(derivative_coeffs, NewParams.to_timestamp(x_point))

    def calc_center_of_mass_with_CP(self):
        # Центр масс (без весов)
        center_of_mass_with_CP = (
            (float(self.model.CP[0]) + self.model.t1[0] + self.model.t2[0] + self.model.t3[0] + self.model.t4[0]) / 5,
            (float(self.model.CP[1]) + self.model.t1[1] + self.model.t2[1] + self.model.t3[1] + self.model.t4[1]) / 5)

        center_of_mass_with_CP_x = center_of_mass_with_CP[0]
        center_of_mass_with_CP_y = center_of_mass_with_CP[1]

        return center_of_mass_with_CP

    def euclidean_distance_center_of_mass_with_CP(self):
        # Вычисление евклидового расстояния
        return sqrt(self.center_of_mass_with_CP[0] ** 2 + self.center_of_mass_with_CP[1] ** 2)


    def calc_center_of_mass_without_weight(self):
        # Центр масс (без весов)
        center_of_mass = (
            (self.model.t1[0] + self.model.t2[0] + self.model.t3[0] + self.model.t4[0]) / 4,
            (self.model.t1[1] + self.model.t2[1] + self.model.t3[1] + self.model.t4[1]) / 4,
        )

        center_of_mass_x = center_of_mass[0]
        center_of_mass_y = center_of_mass[1]
        return center_of_mass

    def euclidean_distance_center_of_mass_without_weight(self):
        # Вычисление евклидового расстояния
        return sqrt(self.center_of_mass[0] ** 2 + self.center_of_mass[1] ** 2)

    def get_components(self):
        # Выберите нужный диапазон данных между t1up и t4up
        data = self.df.loc[self.model.t1[0]:self.model.t4[0], ['high', 'low', 'close', 'open']]

        # Инициализируйте PCA с 4 компонентами
        pca = PCA(n_components=4)

        # Обучите модель на данных
        pca.fit(data)

        # Получите объясненную дисперсию для каждой компоненты
        component_1, component_2, component_3, component_4 = pca.explained_variance_ratio_

        return component_1, component_2, component_3, component_4

    def get_normalized_data(self):
        normalized_data = NewParams.prepare_data(self.df, self.model.t1, self.model.t4)
        self.normalized_data = normalized_data

        return self.normalized_data

    def get_mse_d1_t1_t3(self):
        mse_d1_t1_t3, coeffs_d1_t1_t3, x_data_d1_t1_t3 = NewParams.compute_mse_for_degree(
            self.df, self.normalized_data, self.model.t1, self.model.t3, 1)
        coeffs_d1_t1_t3 = sum(coeffs_d1_t1_t3)
        return mse_d1_t1_t3

    def get_mse_d1_t1_t4(self):
        mse_d1_t1_t4, coeffs_d1_t1_t4_orig, x_data_d1_t1_t4 = NewParams.compute_mse_for_degree(
            self.df, self.normalized_data, self.model.t1, self.model.t4, 1)
        coeffs_d1_t1_t4 = sum(coeffs_d1_t1_t4_orig)

        return mse_d1_t1_t4, x_data_d1_t1_t4

    def get_d2_t1_t4(self):
        mse_d2_t1_t4, coeffs_d2_t1_t4_orig, x_data_d2_t1_t4 = NewParams.compute_mse_for_degree(
            self.df, self.normalized_data, self.model.t1, self.model.t4, 2)
        coeffs_d2_t1_t4 = sum(coeffs_d2_t1_t4_orig)
        return mse_d2_t1_t4, coeffs_d2_t1_t4

    def get_mse_d3_t1_t4(self):
        mse_d3_t1_t4, coeffs_d3_t1_t4_orig, x_data_d3_t1_t4 = NewParams.compute_mse_for_degree(
            self.df, self.normalized_data, self.model.t1, self.model.t4, 3)
        coeffs_d3_t1_t4 = sum(coeffs_d3_t1_t4_orig)
        return mse_d3_t1_t4

    def get_mse_d1_t2_t4(self):
        mse_d1_t2_t4, coeffs_d1_t2_t4, x_data_d1_t2_t4 = NewParams.compute_mse_for_degree(
            self.df, self.normalized_data, self.model.t2, self.model.t4, 1)
        coeffs_d2_t2_t4 = sum(coeffs_d1_t2_t4)
        return mse_d1_t2_t4

    def get_mse_d2_t2_t4(self):
        mse_d2_t2_t4, coeffs_d2_t2_t4, x_data_d2_t2_t4 = NewParams.compute_mse_for_degree(
            self.df, self.normalized_data, self.model.t2, self.model.t4, 2)
        coeffs_d2_t2_t4 = sum(coeffs_d2_t2_t4)
        return mse_d2_t2_t4

    def get_mse_d3_t2_t4(self):
        mse_d3_t2_t4, coeffs_d3_t2_t4, x_data_d3_t2_t4 = NewParams.compute_mse_for_degree(
            self.df, self.normalized_data, self.model.t2, self.model.t4, 3)
        coeffs_d3_t2_t4 = sum(coeffs_d3_t2_t4)
        return coeffs_d3_t2_t4

    def get_mse_d4_t1_t4(self):
        mse_d4_t1_t4, coeffs_d4_t1_t4_orig, x_data_d4_t1_t4 = NewParams.compute_mse_for_degree(
            self.df, self.normalized_data, self.model.t1, self.model.t4, 4)
        coeffs_d4_t1_t4 = sum(coeffs_d4_t1_t4_orig)
        return coeffs_d4_t1_t4_orig

    def model_ridge_regression(self):
        # Выбираем интервал от t1up до t4up и вычисляем среднее между high и low
        interval_data = (self.df.loc[self.model.t1[0]:self.model.t4[0], 'high'] + self.df.loc[
                                                           self.model.t1[0]:self.model.t4[0],
                                                           'low']) / 2

        # Создаём объект Ridge Regression с желаемым параметром регуляризации (например, alpha=1.0)
        ridge_model = Ridge(alpha=1.0)

        # Обучаем модель на ваших данных (X - матрица признаков, y - целевая переменная)
        ridge_model.fit(self.x_data_d1_t1_t4.reshape(-1, 1), interval_data)

        # Предсказываем значения на тех же данных, на которых обучались
        y_predicted = ridge_model.predict(self.x_data_d1_t1_t4.reshape(-1, 1))
        return interval_data, y_predicted

    def skewness_and_kurt(self):
        # Вычисление асимметрии и эксцесса
        skewness, kurt = NewParams.compute_skewness_kurtosis(self.interval_data,
                                                             self.y_predicted)
        return skewness, kurt

    def open_close_statistic(self):
        mean_open_close, std_dev_open_close, skewness_open_close = NewParams.calculate_statistics(
            self.df, ['open', 'close'], self.model.t1[0], self.model.t4[0])
        return skewness_open_close

    def close_statistic(self):
        mean_close, std_dev_close, skewness_close = NewParams.calculate_statistics(self.df, [
            'close'], self.model.t1[0], self.model.t4[0])
        return skewness_close
    def get_derivative_at_x_t1_d4(self):
        x_t1 = self.df.loc[self.model.t1[0], 'dateTime']
        derivative_at_x_t1_d4 = NewParams.compute_derivative(self.coeffs_d4_t1_t4_orig, x_t1)
        return derivative_at_x_t1_d4

    def statistic_between_lines(self):
        # Извлечение и объединение значений 'open', 'close', 'high', 'low' между прямыми
        all_values_between_lines = self.df.loc[self.model.t1[0]:self.model.t4[0],
                                   ['open', 'close', 'high',
                                    'low']].values.flatten()
        kurtosis_distribution = kurtosis(all_values_between_lines)
        return kurtosis_distribution

    def all_params(self):
        skewness_open = self.statistic_params_beetwen_lines()
        center_of_mass_with_CP = self.calc_center_of_mass_with_CP()
        center_of_mass = self.calc_center_of_mass_without_weight()
        euclidean_distance_center_of_mass_with_CP = self.euclidean_distance_center_of_mass_with_CP()
        euclidean_distance_center_of_mass_without_weight = self.euclidean_distance_center_of_mass_without_weight()
        component_1, component_2, component_3, component_4 = self.get_components()
        mse_d2_t1_t4, coeffs_d2_t1_t4 = self.get_d2_t1_t4()
        mse_d3_t1_t4 = self.get_mse_d3_t1_t4()
        skewness, kurt = self.skewness_and_kurt()
        skewness_open_close = self.open_close_statistic()
        mse_d2_t2_t4 = self.get_mse_d2_t2_t4()
        mse_d1_t2_t4 = self.get_mse_d1_t2_t4()
        coeffs_d3_t2_t4 = self.get_mse_d3_t2_t4()
        mse_d1_t1_t3 = self.get_mse_d1_t1_t3()
        skewness_close = self.close_statistic()
        derivative_at_x_t1_d4 = self.get_derivative_at_x_t1_d4()
        kurtosis_distribution = self.statistic_between_lines()
        return (skewness_open,
                center_of_mass_with_CP[0],
                center_of_mass[1],
                euclidean_distance_center_of_mass_with_CP,
                euclidean_distance_center_of_mass_without_weight,
                component_1,
                component_2,
                component_3,
                component_4,
                mse_d2_t1_t4,
                coeffs_d2_t1_t4,
                mse_d3_t1_t4,
                self.mse_d1_t1_t4,
                skewness,
                kurt,
                skewness_open_close,
                mse_d2_t2_t4,
                mse_d1_t2_t4,
                coeffs_d3_t2_t4,
                mse_d1_t1_t3,
                skewness_close,
                derivative_at_x_t1_d4,
                kurtosis_distribution
                )