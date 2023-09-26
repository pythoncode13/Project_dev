import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.ndimage.filters import uniform_filter1d
from scipy.integrate import quad
import math


def distance_between_points(point1, point2):
    return ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5

def angle_between_points(point1, point2, point3):
    a = distance_between_points(point2, point3)
    b = distance_between_points(point1, point3)
    c = distance_between_points(point1, point2)
    cos_angle = (c ** 2 - b ** 2 - a ** 2) / (-2 * a * b)
    return np.degrees(np.arccos(cos_angle))

def length_t1_t3_height_t1_t2_check(t1, t2, t3):
    x_all = [t1[0], t2[0], t3[0]]
    y_all = [t1[1], t2[1], t3[1]]
    cs = CubicSpline(x_all, y_all)
    def speed(x):
        dx = cs(x, 1)
        return (1 + dx ** 2) ** 0.5
    length_t1_t3, _ = quad(speed, t1[0], t3[0])
    height_t1_t2 = t2[1] - t1[1]

    return length_t1_t3, height_t1_t2


def extract_features(df, t1, t2, t3):

    t1_price = df.loc[t1[0], 'close']
    t3_price = df.loc[t3[0], 'close']

    # Срез между t1 и t3
    slice_df = df.loc[t1[0] + 1:t3[0] - 1, 'close']

    # Добавление ключевых точек
    x_all = [t1[0]] + list(slice_df.index) + [t3[0]]
    y_all = [t1_price] + list(slice_df.values) + [t3_price]

    # Создание кубического сплайна
    cs = CubicSpline(x_all, y_all)

    # Генерация новых точек вдоль сплайна
    x_new = np.linspace(min(x_all), max(x_all), 1000)
    y_new = cs(x_new)

    # Подгонка полиномов 2-го и 3-го порядка
    poly2_fit = np.polyfit(x_new, y_new, 2)
    poly3_fit = np.polyfit(x_new, y_new, 3)

    y_poly2_fit = np.polyval(poly2_fit, x_new)
    y_poly3_fit = np.polyval(poly3_fit, x_new)

    ## Вычисление первой производной для полинома второго порядка
    first_derivative_poly2 = np.gradient(y_poly2_fit, x_new)
    # Вычисление первой производной для полинома третьего порядка
    first_derivative_poly3 = np.gradient(y_poly3_fit, x_new)

    # Функция для проверки количества изменений знака
    def check_sign_changes(derivatives):
        changes = np.sign(np.diff(derivatives))
        sign_changes = np.count_nonzero(changes)
        return sign_changes

    # Применение функции к обеим первым производным
    sign_changes_poly2 = check_sign_changes(first_derivative_poly2)
    sign_changes_poly3 = check_sign_changes(first_derivative_poly3)

    # Сглаживание первой производной с использованием скользящего среднего
    window_size = 5
    smoothed_first_derivative_poly2 = uniform_filter1d(first_derivative_poly2, size=window_size)
    smoothed_first_derivative_poly3 = uniform_filter1d(first_derivative_poly3, size=window_size)

    # Разделение на сегменты и анализ направления
    def analyze_segments(derivatives, num_segments=5):
        segment_size = len(derivatives) // num_segments
        directions = []
        for i in range(num_segments):
            segment = derivatives[i * segment_size:(i + 1) * segment_size]
            direction = "восходящий" if np.mean(segment) > 0 else "нисходящий"
            directions.append(direction)
        return directions

    directions_poly2 = analyze_segments(smoothed_first_derivative_poly2)
    directions_poly3 = analyze_segments(smoothed_first_derivative_poly3)

    def is_valid_curve(directions):
        return (
            directions[0] == 'восходящий' and
            directions[1] == 'восходящий' and
            (directions[2] == 'восходящий' or directions[2] == 'нисходящий') and
            directions[3] == 'нисходящий' and
            directions[4] == 'нисходящий'
        )

    valid_poly2 = is_valid_curve(directions_poly2)
    valid_poly3 = is_valid_curve(directions_poly3)

    length_t1_t3, height_t1_t2 = length_t1_t3_height_t1_t2_check(t1, t2, t3)
    distance_t1_t2 = distance_between_points(t1, t2)
    angle_t1_t2_t3 = angle_between_points(t1, t2, t3)

    # Вычисление первой производной
    # cs_prime = cs.derivative(n=1)
    y_prime = np.gradient(y_new, x_new)

    # Вычисление второй производной
    # cs_double_prime = cs.derivative(n=2)
    y_double_prime = np.gradient(y_prime, x_new)

    y_prime_mean = np.mean(y_prime)
    y_double_prime_mean = np.mean(y_double_prime)
    return {
        'length_t1_t3': length_t1_t3,
        'height_t1_t2': height_t1_t2,
        'distance_t1_t2': distance_t1_t2,
        'angle_t1_t2_t3': angle_t1_t2_t3,
        # 'valid_poly2': valid_poly2,
        # 'valid_poly3': valid_poly3,
        # 'directions_poly2': directions_poly2,
        # 'directions_poly3': directions_poly3,
        # 'sign_changes_poly2': sign_changes_poly2,
        # 'sign_changes_poly3': sign_changes_poly3,
        # 'x': x_new,
        # 'y': y_new,
        'y_prime_mean': y_prime_mean,
        'y_double_prime_mean': y_double_prime_mean,
    }
