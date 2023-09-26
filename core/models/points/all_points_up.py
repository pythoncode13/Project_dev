import pandas as pd
import numpy as np
from typing import List, Tuple


def find_t1_up(df: pd.DataFrame) -> List[Tuple[int, float]]:
    """
    Поиск т1.
    Находит бар у которого мин лоу среди предыдущих и последующих 5 баров.
    """
    df['low_min'] = df['low'].rolling(11, center=True).min()
    t1_up_conditions = df['low'] == df['low_min']

    all_t1_up = [(idx, row['low']) for idx, row in df[t1_up_conditions].iterrows()]

    return all_t1_up


def find_t2_up(df: pd.DataFrame) -> List[Tuple[int, float]]:
    """
    Поиск т2.
    Находит бар у которого макс хай среди предыдущих и последующих 2 баров.
    """
    all_t2_up = []

    # Находим условие, где текущий 'high' больше или равен следующему 'high'
    t2_up_conditions = df['high'] >= df['high'].shift(-1)

    # Используем условие для фильтрации и итерации по нужным строкам
    # Проверка на наличие предыдущей t2 с тем же значением 'high'.
    # Если такая есть, пропускаем эту свечу.
    for idx, row in df[t2_up_conditions].iterrows():
        if all_t2_up and all_t2_up[-1][1] == row['high']:
            continue
        all_t2_up.append((idx, row['high']))


    return all_t2_up


def find_t3_up(df: pd.DataFrame) -> List[Tuple[int, float]]:
    """Поиск t3: точек, где текущий 'low' меньше или равен следующему 'low'
    и меньше или равен предыдущему 'low'"""

    # Находим условие, где текущий 'low' меньше или равен следующему 'low'
    condition_next_low = df['low'] <= df['low'].shift(-1)

    # Находим условие, где текущий 'low' меньше или равен предыдущему 'low'
    condition_prev_low = df['low'] <= df['low'].shift(1)

    # Комбинируем оба условия,
    # чтобы найти строки, удовлетворяющие обоим условиям
    t3_conditions = condition_next_low & condition_prev_low

    # Инициализируем список для хранения найденных точек t3
    all_t3_up = []

    # Итерируем по строкам датафрейма,
    # удовлетворяющим условиям, и добавляем их в список
    for idx, row in df[t3_conditions].iterrows():
        # Проверяем на наличие предыдущей t3 с тем же значением 'low'
        # Если такая есть, заменяем её этой свечой
        if all_t3_up and all_t3_up[-1][1] == row['low']:
            all_t3_up[-1] = (idx, row['low'])
        else:
            all_t3_up.append((idx, row['low']))


    return all_t3_up


def find_t4_up(df: pd.DataFrame) -> List[Tuple[int, float]]:
    """Поиск t4: точек, где текущий 'high' больше максимального 'high'
    за предыдущие 3 свечи, и текущий 'high' больше или равен
    следующему 'high', с дополнительными условиями на 'low'."""

    # Инициализируем список для хранения найденных точек t4
    all_t4_up = []

    # Проходим по датафрейму с учетом окон
    # для анализа предыдущих и следующих свечей
    for i in range(3, len(df) - 4):
        curr_candle = df.iloc[i]
        prev_candles = df.iloc[i - 3: i]
        next_candle = df.iloc[i + 1]
        next_next_candle = df.iloc[i + 2]

        # Проверяем, что текущий 'high'
        # больше максимального 'high' за предыдущие 3 свечи
        condition_high_prev = curr_candle['high'] > prev_candles['high'].max()

        # Проверяем, что текущий 'high' больше или равен следующему 'high'
        condition_high_next = curr_candle['high'] >= next_candle['high']

        # Проверяем условия на 'low' свечей
        condition_low = curr_candle['low'] > next_candle['low'] or (
                next_candle['low'] > next_next_candle['low'] and
                next_candle['high'] > next_next_candle['high']) or (
                                curr_candle['low'] > next_next_candle['low']
                        )

        # Если все условия выполняются, добавляем точку в список
        if condition_high_prev and condition_high_next and condition_low:
            all_t4_up.append((i, curr_candle['high']))


    return all_t4_up


def get_combinations(df, range_limit=30):
    """
    Находит все комбинации из t1, t2, t3.

    Parameters
    ----------
    df : pandas.DataFrame
        Оригинальный DataFrame.
    all_t1_up : List[Tuple[int, float]]
        Список точек t1.
    all_t2_up : List[Tuple[int, float]]
        Список точек t2.
    all_t3_up : List[Tuple[int, float]]
        Список точек t3.
    range_limit : int, optional
        Диапазон, в котором ищем t4. По умолчанию 30.

    Returns
    -------
    List[Tuple[Tuple[int, float], Tuple[int, float], Tuple[int, float]]]
        Список комбинаций.
    """

    all_t1_up = find_t1_up(df)
    all_t2_up = find_t2_up(df)
    all_t3_up = find_t3_up(df)

    # Создаем DataFrame для удобства работы
    df_t1 = pd.DataFrame(all_t1_up, columns=['idx', 'price'])
    df_t2 = pd.DataFrame(all_t2_up, columns=['idx', 'price'])
    df_t3 = pd.DataFrame(all_t3_up, columns=['idx', 'price'])

    # Создаем пустой список для хранения результатов
    combinations = []

    # Создаем массивы для удобства работы с NumPy
    t1_array = df_t1.to_numpy()
    t2_array = df_t2.to_numpy()
    t3_array = df_t3.to_numpy()

    for t1 in t1_array:
        # Вычисляем пределы для t3
        start = t1[0]
        end = start + range_limit

        # Выбираем точки t3, которые попадают в диапазон и у которых цена выше чем у t1
        t3_filtered = t3_array[(t3_array[:, 0] >= start) & (t3_array[:, 0] <= end) & (t1[1] < t3_array[:, 1])]

        for t3 in t3_filtered:
            # Выбираем точки t2, которые находятся между t1 и t3
            t2_filtered = t2_array[(t2_array[:, 0] > t1[0]) & (t2_array[:, 0] < t3[0])]

            # Если t2_filtered пуст, пропускаем эту комбинацию
            if t2_filtered.size == 0:
                continue

            # Выбираем точку t2 с наибольшим значением 'price'
            t2 = t2_filtered[np.argmax(t2_filtered[:, 1])]

            if t2[0] == t3[0]:
                continue

            if t2[1] != max(df.loc[t1[0]+1:t3[0]]['high']):
                continue

            # Вычисляем угловой коэффициент прямой между t1 и t3
            slope = (t3[1] - t1[1]) / (t3[0] - t1[0])

            # Вычисляем значения прямой для всех точек между t1 и t3
            line_values = t1[1] + slope * (np.arange(t1[0] + 1, t3[0] + 1) - t1[0])  # add 1 bars to t3

            # Находим минимальную цену в диапазоне t1:t3
            min_price = df.loc[t1[0]+1:t3[0], 'low'].values  # add 3 bars to t3

            # Если хотя бы одна цена меньше соответствующего значения прямой, пропускаем эту комбинацию
            if np.any(min_price < line_values):
                continue

            # Иначе добавляем комбинацию в список
            combinations.append((t1, t2, t3))


    return combinations


def add_t4_to_combinations(df, range_limit=30):
    """
    Добавляет t4 к комбинациям, если они удовлетворяют условиям.

    Parameters
    ----------
    df : pandas.DataFrame
        Оригинальный DataFrame.
    combinations : List[Tuple[Tuple[int, float], Tuple[int, float], Tuple[int, float]]]
        Список комбинаций.
    all_t4_up : List[Tuple[int, float]]
        Список точек t4.
    range_limit : int, optional
        Диапазон, в котором ищем t4. По умолчанию 30.

    Returns
    -------
    List[Tuple[Tuple[int, float], Tuple[int, float], Tuple[int, float], Tuple[int, float]]]
        Список комбинаций с добавленными t4.
    """

    combinations = get_combinations(df, range_limit=30)

    all_t4_up = find_t4_up(df)

    # Создаем DataFrame для удобства работы
    df_t4 = pd.DataFrame(all_t4_up, columns=['idx', 'price'])

    # Создаем новый список для хранения обновленных комбинаций
    new_combinations = []

    # Создаем массив для удобства работы с NumPy
    t4_array = df_t4.to_numpy()

    for combination in combinations:
        t1, t2, t3 = combination

        # Вычисляем пределы для t4
        start = t3[0]
        end = start + range_limit

        # Выбираем точки t4, которые попадают в диапазон и у которых цена выше чем у t3 и t2
        t4_filtered = t4_array[
            (t4_array[:, 0] >= start) &
            (t4_array[:, 0] <= end) &
            (t3[1] < t4_array[:, 1]) &
            (t2[1] < t4_array[:, 1])
        ]

        for t4 in t4_filtered:

            if t3[0] == t4[0]:
                continue

            # Проверяем, что т3 это мин лоу на участке т3-т4
            # (перенес из find_up_model)
            if df.loc[t3[0]:t4[0], 'low'].min() < t3[1]:
                continue

            # Вычисляем угловой коэффициент прямой между t1 и t3
            slope = (t3[1] - t1[1]) / (t3[0] - t1[0])

            # Вычисляем значения прямой для всех точек между t1 и t4
            line_values = t3[1] + slope * (np.arange(t3[0] + 3, t4[0] + 1) - t3[0])  # add 1 bar to t4

            # Находим минимальную цену в диапазоне t1:t4
            min_price = df.loc[t3[0]+3:t4[0], 'low'].values  # add 1 bar to t4

            # Если хотя бы одна цена меньше соответствующего значения прямой, пропускаем эту комбинацию
            if np.any(min_price < line_values):
                continue

            # Вычисляем угловой коэффициент прямой между t2 и t4
            slope = (t4[1] - t2[1]) / (t4[0] - t2[0])

            # Вычисляем значения прямой для всех точек между t2 и t4
            line_values = t2[1] + slope * (
                        np.arange(t2[0] + 1, t4[0] + 1) - t2[
                    0])  # add 1 bar to t4

            # Находим минимальную цену в диапазоне t1:t4
            min_price = df.loc[t2[0] + 1:t4[0],
                        'low'].values  # add 1 bar to t4

            # Если хотя бы одна цена меньше соответствующего значения прямой, пропускаем эту комбинацию
            if np.any(min_price > line_values):
                continue

            # Вычисляем значения прямой для всех точек между t1 и t4
            line_values = t2[1] + slope * (
                    np.arange(t2[0] + 1, t4[0] - 2) - t2[
                0])  # add 1 bar to t4

            # Находим минимальную цену в диапазоне t1:t4
            min_price = df.loc[t2[0] + 1:t4[0] - 3,
                        'high'].values  # add 1 bar to t4

            # Если хотя бы одна цена меньше соответствующего значения прямой, пропускаем эту комбинацию
            if np.any(min_price > line_values):
                continue

            # Добавляем комбинацию в список
            new_combinations.append((t1, t2, t3, t4))

    return new_combinations
