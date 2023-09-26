import numpy as np

from core.models_trade_property.base_trade_analysis_short import \
    BaseTradeAnalysis
from core.models_trade_property.down_model_property import DownModelProperty
from core.models_trade_property.advanced_trade_analysis_short import \
    AdvancedTradeAnalysis
from other_modules.timing_decorator import timing_decorator
from core.point_combinations.treand_models.advanced_model_propetry import AdvancedModelProperty


@timing_decorator
def get_short_setup(df, activated_models_down, ticker):
    """Подготовка данных для сетапа."""

    all_base_setup_parameters = []

    for model in activated_models_down:
        t1 = model.t1
        t2 = model.t2
        t3 = model.t3
        t4 = model.t4

        advanced_property = AdvancedModelProperty(model)
        _, _, mse_t3_t4, _, _ = advanced_property.get_mse

        if mse_t3_t4 < 0.0001:
            continue

        """------------- часть про активацию модели ------------"""
        # Находим лоу каждой свечи
        lower_body_edge = df[['low']].min(axis=1)

        # Задаем диапазон поиска
        lower_limit = int(t4[0] + 1)
        upper_limit = min(int(model.properties.dist_cp_t4_x1), len(df))

        # Находим индексы всех баров в нашем диапазоне,
        # лоу которых закрылся целевого уровня
        bars_below_t4 = np.where(
            lower_body_edge[lower_limit:upper_limit] < t4[1])

        # Если таких баров нет, bars_below_t4 будет пустым
        # и мы пропускаем этот шаг цикла
        if bars_below_t4[0].size == 0:
            continue

        # В противном случае, берем индекс первого такого бара,
        # учитывая смещение на lower_limit
        first_bar_below_t4 = bars_below_t4[0][0] + lower_limit
        """------------- формирование сетапа ------------"""

        entry_index = first_bar_below_t4
        entry_date = df.loc[entry_index, 'dateTime']

        entry_price = t4[1]

        stop_price = t1[1]

        down_take_200 = t4[1] - (t1[1] - t4[1]) * 2

        take_price = t4[1] - 0.9 * (t4[1] - down_take_200)

        analysis_base = BaseTradeAnalysis(t1,
                                          t2,
                                          t3,
                                          entry_price,
                                          stop_price,
                                          take_price)

        """------------- свойства ------------"""
        '''Получаем дополнительные свойства модели.'''

        # Создаем объект для расчетов дополнительных
        # свойств модели
        model_property = DownModelProperty(df, t4, analysis_base)

        # Геометрические расчеты
        angle_deg_LT, _, _, _ = model_property.get_LT_down
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
         ) = analysis_advanced.calculate_param_t2_t3_down_enter

        base_setup_parameters = (
            entry_date,
            ticker,
            entry_index,
            entry_price,
            stop_price,
            take_price,
            analysis_base.stop_percent_difference,
            analysis_base.take_percent_difference,
        )

        advanced_setup_parameters = (
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
            *advanced_property.get_mse
        )

        all_setup_parameters = (base_setup_parameters, advanced_setup_parameters)
        all_base_setup_parameters.append(all_setup_parameters)

    return all_base_setup_parameters
