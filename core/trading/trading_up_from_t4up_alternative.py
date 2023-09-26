import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from core.models_trade_property.base_trade_analysis_long import \
    BaseTradeAnalysis
from core.models_trade_property.up_model_property import UpModelProperty
from core.models_trade_property.advanced_trade_analysis_long import \
    AdvancedTradeAnalysis
from other_modules.timing_decorator import timing_decorator
from core.point_combinations.treand_models.advanced_model_propetry import AdvancedModelProperty
from core.model_utilities.harmonica_params import HarmonicaParams
from core.model_utilities.fan_line_v0 import FanLine
from core.model_utilities.line import Line
from core.model_utilities.point import Point
from core.model_utilities.new_params import NewParams

@timing_decorator
def prepare_trading_setup(df, activated_models_up, ticker):
    """Подготовка данных для сетапа."""

    all_base_setup_parameters = []

    for model in activated_models_up:
        t1up = model.t1
        t2up = model.t2
        t3up = model.t3
        t4up = model.t4

        # if t4up[0] != 60:
        #     continue

        """------------- часть про активацию модели ------------"""
        # Находим нижний край тела каждой свечи
        lower_body_edge = df[['high']].min(axis=1)

        # Задаем диапазон поиска
        lower_limit = int(t4up[0] + 1)
        upper_limit = min(int(model.properties.dist_cp_t4_x1), len(df))

        # Находим индексы всех баров в нашем диапазоне,
        # нижний край которых закрылся выше уровня model.up_take_lines[1]
        bars_above_t4up = np.where(
            lower_body_edge[lower_limit:upper_limit] > t4up[1])

        # Если таких баров нет, bars_above_t4up будет пустым
        # и мы пропускаем этот шаг цикла
        if bars_above_t4up[0].size == 0:
            continue

        # В противном случае, берем индекс первого такого бара,
        # учитывая смещение на lower_limit
        first_bar_above_t4up = bars_above_t4up[0][0] + lower_limit
        """------------- формирование сетапа ------------"""
        x_CP = float(model.CP[0])
        lower_limit = max(0, int(x_CP - (t4up[0] - x_CP) * 3))
        #TODO вернуть 110
        upper_limit = min(int(first_bar_above_t4up + 110), len(df))
        sliced_df = df.iloc[lower_limit:upper_limit]


        #
        # # Выбираем все столбцы кроме 'dateTime'
        # columns_to_normalize = [col for col in df.columns if col != 'dateTime']
        # df_to_normalize = df[columns_to_normalize]
        #
        # # Нормализуем выбранные столбцы
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # normalized_array = scaler.fit_transform(df_to_normalize)
        #
        # # Преобразуем нормализованные данные обратно в DataFrame
        # normalized_df = pd.DataFrame(normalized_array,
        #                              columns=columns_to_normalize,
        #                              index=df.index)
        #
        # # Объединяем с столбцом 'dateTime'
        # normalized_df = pd.concat([df['dateTime'], normalized_df], axis=1)

        advanced_property = AdvancedModelProperty(model, sliced_df)
        _, _, mse_t3_t4, _, _ = advanced_property.get_mse

        if mse_t3_t4 < 0.0001:
            continue

        entry_index = first_bar_above_t4up
        entry_date = sliced_df.loc[entry_index, 'dateTime']

        entry_price = t4up[1]

        stop_price = t1up[1]

        take_price = t4up[1] + 0.9 * (
                model.properties.up_take_200 - t4up[1])
        # take_price = t4up[1] + 0.9 * (
        #         model.properties.up_take_100 - t4up[1])
        analysis_base = BaseTradeAnalysis(t1up,
                                          t2up,
                                          t3up,
                                          entry_price,
                                          stop_price,
                                          take_price)

        """------------- свойства ------------"""
        '''Получаем дополнительные свойства модели.'''

        # Создаем объект для расчетов дополнительных
        # свойств модели
        up_enter_point1 = (t3up[0] + 1, t2up[1] * 1.2)
        model_property = UpModelProperty(sliced_df, up_enter_point1,
                                         analysis_base)

        # Геометрические расчеты
        angle_deg_LT, _, _, _ = model_property.get_LT_up
        max_distance, _ = model_property.find_point_t1_t2

        '''На основе предыдущих шагов производим расчеты,
        для получения дополнительных параметров.'''
        # Создаем объект расчета дополнительных параметров
        analysis_advanced = AdvancedTradeAnalysis(
            model_property)

        # Получаем различные статистические данные,
        # геометрические измерения
        (percentage_difference_min_low_and_t1,
         percentage_difference_min_close_and_t1,
         percentage_difference_max_close_and_t1,
         diff_price_change, length3_point_t1_t2_norm,
         angle_t2_t3, area_under_curve
         ) = analysis_advanced.calculate_property

        (angle_t3_enter,
         radius_curvature_t2_t3_enter
         ) = analysis_advanced.calculate_param_t2_t3_up_enter

        """------------- взаимодействие с веером ------------"""
        fanline = FanLine(model, entry_index)

        l0_line, lt1_line, lc1_line = fanline.build_fan()

        # Точка входа выше или ниже л0
        open_value = sliced_df.loc[entry_index, 'open']
        line_value_at_candle = l0_line.slope * entry_index + l0_line.intercept

        if open_value > line_value_at_candle:
            entry_above_l0 = 1
        elif open_value < line_value_at_candle:
            entry_above_l0 = 0
        else:
            entry_above_l0 = 2

        # Точка входа выше или ниже л0
        high_value = sliced_df.loc[entry_index, 'high']
        line_value_at_candle_high = lc1_line.slope * entry_index + lc1_line.intercept

        if high_value > line_value_at_candle_high:
            entry_above_lc1_line = 1
        elif high_value < line_value_at_candle_high:
            entry_above_lc1_line = 0
        else:
            entry_above_lc1_line = 2

        # Количество свечей на верхнем и нижнем наружных областей

        # Вычисляем значение прямой для каждого индекса в диапазоне между start_index и end_index
        line_values = lc1_line.slope * sliced_df.loc[model.t1[0]:model.t4[
            0]].index + lc1_line.intercept

        # Получаем все значения 'high' в этом диапазоне
        high_values = sliced_df.loc[model.t1[0]:model.t4[0], 'high']

        # Находим те, которые выше прямой
        high_above_lc1 = high_values[high_values > line_values]
        # Получаем количество свечей
        num_high_above_lc1 = (high_above_lc1.shape[0] * 0.5)

        # Получаем все значения 'high' в этом диапазоне
        close_values = sliced_df.loc[model.t1[0]:model.t4[0], 'close']

        # Находим те, которые выше прямой
        close_above_lc1 = close_values[close_values > line_values]
        # Получаем количество свечей
        num_close_above_lc1 = close_above_lc1.shape[0]

        # Вычисляем значение прямой для каждого индекса в диапазоне между start_index и end_index
        line_values_LT = lt1_line.slope * sliced_df.loc[model.t1[0]:model.t4[
            0]].index + lt1_line.intercept

        # Получаем все значения 'high' в этом диапазоне
        low_values = sliced_df.loc[model.t1[0]:model.t4[0], 'low']

        # Находим те, которые выше прямой
        low_below_lt1 = low_values[low_values < line_values_LT]
        # Получаем количество свечей
        num_low_below_lt1 = (low_below_lt1.shape[0] * 0.5)

        # Получаем все значения 'high' в этом диапазоне
        close_values_below = sliced_df.loc[model.t1[0]:model.t4[0], 'close']

        # Находим те, которые выше прямой
        close_below_lt1 = close_values_below[
            close_values_below < line_values_LT]
        # Получаем количество свечей
        num_close_below_lt1 = close_below_lt1.shape[0]

        ratio_high_low_fan = round((num_high_above_lc1 / num_low_below_lt1), 3)
        if num_close_below_lt1 == 0:
            ratio_close_fan = 0
        else:
            ratio_close_fan = round((num_close_above_lc1 / num_close_below_lt1), 3)

        ratio_sum_fan = round(((num_high_above_lc1 + num_close_above_lc1)
                               / (num_low_below_lt1 + num_close_below_lt1)), 3)

        dist_t1_t4 = model.t4[0] - model.t1[0]

        ratio_num_high_above_lc1_to_all = round(
            (num_high_above_lc1 / dist_t1_t4), 3)
        ratio_num_low_below_lt1_to_all = round(
            (num_low_below_lt1 / dist_t1_t4), 3)
        ratio_num_close_above_lc1_to_all = round(
            (num_close_above_lc1 / dist_t1_t4), 3)
        ratio_num_close_below_lt1_to_all = round(
            (num_close_below_lt1 / dist_t1_t4), 3)

        high_low_in_medium_sector = (dist_t1_t4 / 2) - (
                    num_high_above_lc1 - num_low_below_lt1)
        close_in_medium_sector = dist_t1_t4 - (
                    num_close_above_lc1 + num_close_below_lt1)

        sum_in_medium_sector = (
                    high_low_in_medium_sector + close_in_medium_sector)
        sum_in_upper_sector = (num_high_above_lc1 + num_close_above_lc1)
        sum_in_lower_sector = (num_low_below_lt1 + num_close_below_lt1)

        ratio_medium_to_lower = round(
            (sum_in_medium_sector / sum_in_lower_sector), 3)
        ratio_medium_to_upper = round(
            (sum_in_medium_sector / sum_in_upper_sector), 3)
        ratio_medium_to_lateral = round((sum_in_medium_sector / (
                    sum_in_lower_sector + sum_in_upper_sector)), 3)
        ratio_medium_to_all = round((sum_in_medium_sector / dist_t1_t4), 3)

        t3_t4_line = Line.calculate((float(model.t3[0]), float(model.t3[1])),
                                    (float(model.t4[0]), float(model.t4[1]))
                                    )
        t3_t4_point = (float(model.t3[0]), float(model.t4[1]))

        CP_t3_t4_line = Line.calculate(
            (float(model.CP[0]), float(model.CP[1])),
            t3_t4_point
        )

        # distances = fanline.dist_lines()
        # distances_tuple = tuple(distances.values())
        # print(distances_tuple)

        params = NewParams(sliced_df, model)

        # force_close_minutes = (model.properties.dist_cp_t4_x2 - model.t4[
        #     0]) * 3
        skewness_open = params.all_params
        base_setup_parameters = (
            entry_date,
            ticker,
            entry_index,
            entry_price,
            stop_price,
            take_price,
            np.round(analysis_base.stop_percent_difference, 4),
            np.round(analysis_base.take_percent_difference, 4),
        )

        advanced_setup_parameters = tuple(map(lambda x: np.round(float(x), 4),
                                              (
            # percentage
            diff_price_change,
            percentage_difference_min_low_and_t1,
            percentage_difference_min_close_and_t1,
            percentage_difference_max_close_and_t1,

            # std_dev_y_mean_y
            *analysis_advanced.compute_std_dev_y_mean_y,

            # Вычисляем соотношение хвоста и тела свечи
            # для различных точек
            *analysis_advanced.candle_tail_body_parameters,

            # geometric
            area_under_curve,
            radius_curvature_t2_t3_enter,
            angle_t2_t3,
            angle_t3_enter,
            angle_deg_LT,
            max_distance,
            length3_point_t1_t2_norm,

            # indicators
            *analysis_advanced.get_rsi,
            *analysis_advanced.get_vwap,

            model.properties.model_interval,

            *advanced_property.calculate_golden_ratio,

            model.properties.CP_to_t1,
            *advanced_property.nums_of_distances_to_points(direction='up_model'),
            *advanced_property.get_mse,
            # *HarmonicaParams(model).calculate_distances_and_ratios(),
            model.parallel,
            ###################
            entry_above_l0,
            entry_above_lc1_line,

            ratio_num_high_above_lc1_to_all,
            ratio_num_low_below_lt1_to_all,
            ratio_num_close_above_lc1_to_all,
            ratio_num_close_below_lt1_to_all,
            ratio_high_low_fan,
            ratio_close_fan,
            ratio_sum_fan,
            ratio_medium_to_lower,
            ratio_medium_to_upper,
            ratio_medium_to_lateral,
            ratio_medium_to_all,

            *fanline.angle_fan(),
            # *fanline.build_quadro(),
            # *distances_tuple
            *params.all_params(),
            # force_close_minutes
        )))

        all_setup_parameters = (base_setup_parameters, advanced_setup_parameters, model)
        all_base_setup_parameters.append(all_setup_parameters)

    return all_base_setup_parameters
