import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.ndimage.filters import uniform_filter1d
from scipy.integrate import quad
import math

# from core.loader_binance import LoaderBinance
#
# # Загрузка данных
# ticker = 'BTCUSDT'
# timeframe = '30MINUTE'
# s_date = "2023-08-01"
# u_date = "2023-08-05"
#
# loader = LoaderBinance(ticker, timeframe, s_date, u_date)
# loader.get_historical_data()
# loader.add_indicators()
# df = loader.df
#
# # Определение ключевых точек
# t1 = (98.0, 28968.0)
# t3 = (107.0, 29078.43)


def combinated_validate(df, t1, t2, t3):

    # t1_price = df.loc[t1[0], 'close']
    # t3_price = df.loc[t3[0], 'close']
    #
    # # Срез между t1 и t3
    # slice_df = df.loc[t1[0] + 1:t3[0] - 1, 'close']
    #
    # # Добавление ключевых точек
    # x_all = [t1[0]] + list(slice_df.index) + [t3[0]]
    # y_all = [t1_price] + list(slice_df.values) + [t3_price]

    # Срез между t1 и t3
    slice_df = df.loc[t1[0] + 1:t3[0] - 1, 'high']

    # Добавление ключевых точек
    x_all = [t1[0]] + list(slice_df.index) + [t3[0]]
    y_all = [t1[1]] + list(slice_df.values) + [t3[1]]

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

    print(f"Количество изменений знака для полинома второго порядка: {sign_changes_poly2}")
    print(f"Количество изменений знака для полинома третьего порядка: {sign_changes_poly3}")

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

    print(f"Направления для полинома второго порядка: {directions_poly2}")
    print(f"Направления для полинома третьего порядка: {directions_poly3}")

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

    print(f"Валидность для полинома второго порядка: {valid_poly2}")
    print(f"Валидность для полинома третьего порядка: {valid_poly3}")


    if valid_poly2 and valid_poly3:
        return True

    else:
        return False

    # # Отображение сглаженных первых производных
    # plt.figure(figsize=[10, 6])
    # plt.plot(x_new, smoothed_first_derivative_poly2, '--', label='Сглаженная первая производная (полином 2-го порядка)')
    # plt.plot(x_new, smoothed_first_derivative_poly3, ':', label='Сглаженная первая производная (полином 3-го порядка)')
    # plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    # plt.xlabel('X')
    # plt.ylabel('Сглаженная первая производная')
    # plt.legend(loc='best')
    # plt.title('Сглаживание и анализ сегментов')
    # plt.show()

def length_t1_t3_height_t1_t2_chek(t1, t2, t3):
    # Определение всех точек для сплайна

    x_all = [t1[0], t2[0], t3[0]]
    y_all = [t1[1], t2[1], t3[1]]

    # Создание кубического сплайна
    cs = CubicSpline(x_all, y_all)

    # Функция для вычисления скорости движения по кривой (первая производная сплайна)
    def speed(x):
        dx = cs(x, 1)
        return (1 + dx ** 2) ** 0.5

    # Генерация новых точек вдоль сплайна
    x_new = np.linspace(min(x_all), max(x_all), 1000)
    y_new = cs(x_new)

    # Вычисление производной (скорости движения по кривой) на новых точках
    y_speed = speed(x_new)
    # Численное интегрирование для вычисления длины кривой
    length_t1_t3, _ = quad(speed, t1[0], t3[0])

    # Высота между t1 и t2
    height_t1_t2 = t2[1] - t1[1]


    return length_t1_t3, height_t1_t2