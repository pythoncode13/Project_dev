# Импорты
import pandas as pd
import numpy as np

from collections import namedtuple
# Константы
# Длина участка т1-т3
RANGE_LIMIT_t1_t3 = 30
# Длина участка т3-т4
RANGE_LIMIT_t3_t4 = 30

Combination = namedtuple('Combination', ['t1', 't2', 't3', 't4'])


class DownTrendCombinations:
    def __init__(self,
                 df: pd.DataFrame,
                 range_limit_t1_t3=RANGE_LIMIT_t1_t3,
                 range_limit_t3_t4=RANGE_LIMIT_t3_t4):
        self.df = df
        self.range_limit_t1_t3 = range_limit_t1_t3
        self.range_limit_t3_t4 = range_limit_t3_t4

    # # Вспомогательные функции
    # def _helper_function(self):
    #     pass

    # Основные функции
    def find_t1_down(self):
        """
        Поиск т1.
        Находит бар у которого max high среди предыдущих и последующих 5 баров.
        """
        self.df['high_max'] = self.df['high'].rolling(11, center=True).max()
        t1_down_conditions = self.df['high'] == self.df['high_max']

        all_t1_down = [(idx, row['high_max']) for idx, row in
                     self.df[t1_down_conditions].iterrows()]

        return all_t1_down

    def find_t2_down(self) -> list[tuple[int, float]]:
        """
        Поиск т2.
        Находит бар у которого min low среди предыдущих и последующих 2 баров.
        """
        all_t2_down = []

        # Находим условие, где текущий 'low' больше или
        # равен следующему 'low'
        t2_down_conditions = self.df['low'] <= self.df['low'].shift(-1)

        # Используем условие для фильтрации и итерации по нужным строкам
        # Проверка на наличие предыдущей t2 с тем же значением 'high'.
        # Если такая есть, пропускаем эту свечу.
        for idx, row in self.df[t2_down_conditions].iterrows():
            if all_t2_down and all_t2_down[-1][1] == row['low']:
                continue
            all_t2_down.append((idx, row['low']))

        return all_t2_down

    def find_t3_down(self) -> list[tuple[int, float]]:
        """Поиск t3: точек, где текущий 'high' меньше
        или равен следующему 'high' и меньше или равен предыдущему 'high'"""

        # Находим условие, где текущий 'high' больше или равен следующему 'high'
        condition_next_high = self.df['high'] >= self.df['high'].shift(-1)

        # Находим условие, где текущий 'low' больше или равен предыдущему 'low'
        condition_prev_high = self.df['high'] >= self.df['high'].shift(1)

        # Комбинируем оба условия,
        # чтобы найти строки, удовлетворяющие обоим условиям
        t3_conditions = condition_next_high & condition_prev_high

        # Инициализируем список для хранения найденных точек t3
        all_t3_down = []

        # Итерируем по строкам датафрейма,
        # удовлетворяющим условиям, и добавляем их в список
        for idx, row in self.df[t3_conditions].iterrows():
            # Проверяем на наличие предыдущей t3 с тем же значением 'high'
            # Если такая есть, заменяем её этой свечой
            if all_t3_down and all_t3_down[-1][1] == row['high']:
                all_t3_down[-1] = (idx, row['high'])
            else:
                all_t3_down.append((idx, row['high']))

        return all_t3_down

    def find_t4_down(self) -> list[tuple[int, float]]:
        """Поиск t4: точек, где текущий 'low' меньше минимального 'low'
        за предыдущие 3 свечи, и текущий 'low' меньше или равен
        следующему 'low', с дополнительными условиями на 'high'."""

        # Инициализируем список для хранения найденных точек t4
        all_t4_down = []

        # Проходим по датафрейму с учетом окон
        # для анализа предыдущих и следующих свечей
        for i in range(3, len(self.df) - 4):
            curr_candle = self.df.iloc[i]
            prev_candles = self.df.iloc[i - 3: i]
            next_candle = self.df.iloc[i + 1]
            next_next_candle = self.df.iloc[i + 2]
            # t4_plus_3 = self.df.iloc[i + 3]

            # Проверяем, что текущий 'low'
            # меньше минимального 'low' за предыдущие 3 свечи
            condition_low_prev = curr_candle['low'] < prev_candles[
                'low'].min()

            # Проверяем, что текущий 'low' меньше или равен следующему 'low'
            condition_low_next = curr_candle['low'] <= next_candle['low']

            # Проверяем условия на 'high' свечей
            condition_high = (
                # 1 лоу т4 > лоу след. свечи и
                    (curr_candle['high'] < next_candle['high'])
                    # 2 хай т4+1 < хай т4+2 и лоу т4+1 > лоу т4+2
                    or (next_candle['high'] < next_next_candle['high']
                        and next_candle['low'] > next_next_candle['low'])
                    # 3 хай т4 < хай т4+2
                    or (curr_candle['high'] < next_next_candle['high'])
                    # # 4 хай т4+2 < хай т4+3 и лоу т4 == мин лоу до т4+3
                    # or (next_next_candle['high'] < t4_plus_3['high']
                    #     and curr_candle['low'] == self.df.loc[i:i + 3,
                    #                               'low'].min()
                    #     )
            )

            # Если все условия выполняются, добавляем точку в список
            if condition_low_prev and condition_low_next and condition_high:
                all_t4_down.append((i, curr_candle['low']))

        return all_t4_down

    def get_combinations(self):
        """
        Находит все комбинации из t1, t2, t3.

        Parameters (данные, которые использует функция)
        ----------
        df : pandas.DataFrame
            Оригинальный DataFrame.
        all_t1_down : List[Tuple[int, float]]
            Список точек t1.
        all_t2_down : List[Tuple[int, float]]
            Список точек t2.
        all_t3_down : List[Tuple[int, float]]
            Список точек t3.
        range_limit : int, optional
            Диапазон, в котором ищем t4. По умолчанию 30.

        Returns
        -------
        List[Tuple[Tuple[int, float], Tuple[int, float], Tuple[int, float]]]
            Список комбинаций.
        """

        all_t1_down = self.find_t1_down()
        all_t2_down = self.find_t2_down()
        all_t3_down = self.find_t3_down()

        # Создаем DataFrame для удобства работы
        df_t1 = pd.DataFrame(all_t1_down, columns=['idx', 'price'])
        df_t2 = pd.DataFrame(all_t2_down, columns=['idx', 'price'])
        df_t3 = pd.DataFrame(all_t3_down, columns=['idx', 'price'])

        # Создаем пустой список для хранения результатов
        combinations = []

        # Создаем массивы для удобства работы с NumPy
        t1_array = df_t1.to_numpy()
        t2_array = df_t2.to_numpy()
        t3_array = df_t3.to_numpy()

        for t1 in t1_array:
            # Вычисляем пределы для t3
            start = t1[0]
            end = start + self.range_limit_t1_t3

            # Выбираем точки t3, которые попадают в диапазон
            # и у которых цена ниже чем у t1
            t3_filtered = t3_array[
                (t3_array[:, 0] >= start) & (t3_array[:, 0] <= end) & (
                            t1[1] > t3_array[:, 1])]

            for t3 in t3_filtered:
                # Выбираем точки t2, которые находятся между t1 и t3
                t2_filtered = t2_array[
                    (t2_array[:, 0] > t1[0]) & (t2_array[:, 0] < t3[0])]

                # Если t2_filtered пуст, пропускаем эту комбинацию
                if t2_filtered.size == 0:
                    continue

                # Выбираем точку t2 с наименьшим значением 'price'
                t2 = t2_filtered[np.argmin(t2_filtered[:, 1])]

                if t2[0] == t3[0]:
                    continue

                if t2[1] != min(self.df.loc[t1[0] + 1:t3[0]]['low']):
                    continue

                # Проверяем, что т3 это max high на участке т2-т3
                if self.df.loc[t2[0]:t3[0], 'high'].max() > t3[1]:
                    continue

                # Вычисляем угловой коэффициент прямой между t1 и t3
                slope = (t3[1] - t1[1]) / (t3[0] - t1[0])

                # Вычисляем значения прямой для всех точек между t1 и t3
                line_values = (
                        t1[1] + slope
                        * (np.arange(t1[0] + 1, t3[0] + 1) - t1[0])
                )  # add 1 bars to t3

                # Находим max price в диапазоне t1:t3
                max_price = (
                    self.df.loc[t1[0] + 1:t3[0], 'high'].values
                )  # add 3 bars to t3

                # Если хотя бы одна цена меньше соответствующего
                # значения прямой, пропускаем эту комбинацию
                if np.any(max_price > line_values):
                    continue

                # Иначе добавляем комбинацию в список
                combinations.append((t1, t2, t3))

        return combinations

    def add_t4_to_combinations(self):
        """
        Добавляет t4 к комбинациям, если они удовлетворяют условиям.

        Parameters (данные, которые использует функция)
        ----------
        df : pandas.DataFrame
            Оригинальный DataFrame.
        combinations : List[Tuple[Tuple[int, float], Tuple[int, float],
        Tuple[int, float]]]
            Список комбинаций.
        all_t4_down : List[Tuple[int, float]]
            Список точек t4.
        range_limit : int, optional
            Диапазон, в котором ищем t4. По умолчанию 30.

        Returns
        -------
        List[Tuple[Tuple[int, float], Tuple[int, float], Tuple[int, float],
        Tuple[int, float]]]
            Список комбинаций с добавленными t4.
        """

        combinations = self.get_combinations()

        all_t4_down = self.find_t4_down()

        # Создаем DataFrame для удобства работы
        df_t4 = pd.DataFrame(all_t4_down, columns=['idx', 'price'])

        # Создаем новый список для хранения обновленных комбинаций
        new_combinations = []

        # Создаем массив для удобства работы с NumPy
        t4_array = df_t4.to_numpy()

        for combination in combinations:
            t1, t2, t3 = combination

            # Вычисляем пределы для t4
            start = t3[0]
            end = start + self.range_limit_t3_t4

            # Выбираем точки t4, которые попадают в диапазон и у которых цена
            # ниже чем у t3 и t2
            t4_filtered = t4_array[
                (t4_array[:, 0] >= start) &
                (t4_array[:, 0] <= end) &
                (t3[1] > t4_array[:, 1]) &
                (t2[1] > t4_array[:, 1])
                ]

            for t4 in t4_filtered:

                if t3[0] == t4[0]:
                    continue

                # Вычисляем угловой коэффициент прямой между t1 и t3
                slope = (t3[1] - t1[1]) / (t3[0] - t1[0])

                # Вычисляем значения прямой для всех точек между t1 и t4
                line_values = (
                        t3[1] + slope
                        * (np.arange(t3[0] + 3, t4[0] + 1) - t3[0])
                )  # add 1 bar to t4

                # Находим максимальную цену в диапазоне t1:t4
                max_price = (
                    self.df.loc[t3[0] + 3:t4[0], 'high'].values
                )  # add 1 bar to t4

                # Если хотя бы одна цена больше соответствующего значения
                # прямой, пропускаем эту комбинацию
                if np.any(max_price > line_values):
                    continue

                # Вычисляем угловой коэффициент прямой между t2 и t4
                slope = (t4[1] - t2[1]) / (t4[0] - t2[0])

                # Вычисляем значения прямой для всех точек между t2 и t4
                line_values = (
                        t2[1] + slope
                        * (np.arange(t2[0] + 1, t4[0] + 1) - t2[0])
                )  # add 1 bar to t4

                # Находим максимальную цену в диапазоне t1:t4
                max_price = (
                    self.df.loc[t2[0] + 1:t4[0], 'high'].values
                )  # add 1 bar to t4

                # Если хотя бы одна цена больше соответствующего значения
                # прямой, пропускаем эту комбинацию
                if np.any(max_price < line_values):
                    continue

                # Вычисляем значения прямой для всех точек между t1 и t4
                line_values = (
                        t2[1] + slope
                        * (np.arange(t2[0] + 1, t4[0] - 2) - t2[0])
                )  # add 1 bar to t4

                # Находим минимальную цену в диапазоне t1:t4
                min_price = (
                    self.df.loc[t2[0] + 1:t4[0] - 3, 'low'].values
                )  # add 1 bar to t4

                # Если хотя бы одна цена меньше соответствующего значения
                # прямой, пропускаем эту комбинацию
                if np.any(min_price < line_values):
                    continue

                # Проверяем, что т3 это макс хай на участке т3-т4
                if self.df.loc[t3[0]:t4[0], 'high'].max() > t3[1]:
                    continue
                # # Проверяем, есть ли другие точки t4 между t2 и текущей t4
                # if np.any((t4_array[:, 0] > t2[0]) & (t4_array[:, 0] < t4[0])):
                #     continue

                # Добавляем комбинацию в список
                new_combinations.append(Combination(t1, t2, t3, t4))

        return new_combinations
