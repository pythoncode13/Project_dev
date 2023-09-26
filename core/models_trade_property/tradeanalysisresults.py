from core.models_trade_property.advanced_trade_analysis_long import \
    AdvancedTradeAnalysis


class TradeAnalysisResults:
    def __init__(self, model_property):

        self.model_property = model_property

        # Получаем различные статистические данные,
        # геометрические измерения
        self.percentage_difference_min_low_and_t1 = None
        self.percentage_difference_min_close_and_t1 = None
        self.percentage_difference_max_close_and_t1 = None
        self.diff_price_change = None 
        self.length3_point_t1_t2_norm = None
        self.angle_t2_t3 = None 
        self.area_under_curve = None
        
        
        self.angle_t3_enter = None
        self.radius_curvature_t2_t3_enter = None
        
        
        # Вычисляем соотношение хвоста и тела свечи
        # для различных точек
        self.tail_to_body_ratio_t1 = None
        self.tail_to_body_ratio_t2 = None
        self.tail_to_body_ratio_t3 = None
        self.tail_to_body_ratio_enter_point_back_1 = None
        
        
        # Рассчитываем std_dev_y_mean_y
        self.std_dev_y_mean_y = None
        self.std_dev_y_mean_y_1 = None
        self.std_dev_y_mean_y_2 = None
        self.std_dev_y_mean_y_3 = None
        self.std_dev_y_mean_y_4 = None
        self.std_dev_y_mean_y_5 = None
        self.std_dev_y_mean_y_6 = None
        self.std_dev_y_mean_y_7 = None
        self.std_dev_y_t2_t3_up_enter = None
        
        
        # Получаем значения индикаторов,
        # делаем расчеты на их основе
        # RSI
        self.rsi_value1 = None
        self.rsi_value2 = None
        self.rsi_value3 = None
        self.rsi_value_enter = None
        
        # # VWAP
        self.vwap_t1 = None
        self.vwap_enter = None
        self.vwap_ratio_t1 = None
        self.vwap_ratio_enter = None
        self.vwap_t1_v2 = None
        self.vwap_enter_v2 = None
        self.vwap_ratio_t1_v2 = None
        self.vwap_ratio_enter_v2 = None

    def calculate_properties(self):
        analysis_advanced = AdvancedTradeAnalysis(
            self.model_property)

        # Получаем различные статистические данные,
        # геометрические измерения
        (self.percentage_difference_min_low_and_t1,
         self.percentage_difference_min_close_and_t1,
         self.percentage_difference_max_close_and_t1,
         self.diff_price_change, self.length3_point_t1_t2_norm,
         self.angle_t2_t3, self.area_under_curve
         ) = analysis_advanced.calculate_property

        (self.angle_t3_enter,
         self.radius_curvature_t2_t3_enter
         ) = analysis_advanced.calculate_param_t2_t3_up_enter

        # Вычисляем соотношение хвоста и тела свечи
        # для различных точек
        (self.tail_to_body_ratio_t1,
         self.tail_to_body_ratio_t2,
         self.tail_to_body_ratio_t3,
         self.tail_to_body_ratio_enter_point_back_1
         ) = analysis_advanced.candle_tail_body_parameters

        # Рассчитываем std_dev_y_mean_y
        (self.std_dev_y_mean_y,
         self.std_dev_y_mean_y_1,
         self.std_dev_y_mean_y_2,
         self.std_dev_y_mean_y_3,
         self.std_dev_y_mean_y_4,
         self.std_dev_y_mean_y_5,
         self.std_dev_y_mean_y_6,
         self.std_dev_y_mean_y_7,
         self.std_dev_y_t2_t3_up_enter
         ) = analysis_advanced.compute_std_dev_y_mean_y

        # Получаем значения индикаторов,
        # делаем расчеты на их основе
        # RSI
        (self.rsi_value1,
         self.rsi_value2,
         self.rsi_value3,
         self.rsi_value_enter
         ) = analysis_advanced.get_rsi

        # # VWAP
        (self.vwap_t1,
         self.vwap_enter,
         self.vwap_ratio_t1,
         self.vwap_ratio_enter,
         self.vwap_t1_v2,
         self.vwap_enter_v2,
         self.vwap_ratio_t1_v2,
         self.vwap_ratio_enter_v2,
         ) = analysis_advanced.get_vwap

    def get_all_parameters(self):
        return (
            self.percentage_difference_min_low_and_t1,
            self.percentage_difference_min_close_and_t1,
            self.percentage_difference_max_close_and_t1,
            self.diff_price_change,
            self.length3_point_t1_t2_norm,
            self.angle_t2_t3,
            self.area_under_curve,

            self.angle_t3_enter,
            self.radius_curvature_t2_t3_enter,

            # Вычисляем соотношение хвоста и тела свечи
            # для различных точек
            self.tail_to_body_ratio_t1,
            self.tail_to_body_ratio_t2,
            self.tail_to_body_ratio_t3,
            self.tail_to_body_ratio_enter_point_back_1,

            # Рассчитываем std_dev_y_mean_y
            self.std_dev_y_mean_y,
            self.std_dev_y_mean_y_1,
            self.std_dev_y_mean_y_2,
            self.std_dev_y_mean_y_3,
            self.std_dev_y_mean_y_4,
            self.std_dev_y_mean_y_5,
            self.std_dev_y_mean_y_6,
            self.std_dev_y_mean_y_7,
            self.std_dev_y_t2_t3_up_enter,

            # Получаем значения индикаторов,
            # делаем расчеты на их основе
            # RSI
            self.rsi_value1,
            self.rsi_value2,
            self.rsi_value3,
            self.rsi_value_enter,

            # # VWAP
            self.vwap_t1,
            self.vwap_enter,
            self.vwap_ratio_t1,
            self.vwap_ratio_enter,
            self.vwap_t1_v2,
            self.vwap_enter_v2,
            self.vwap_ratio_t1_v2,
            self.vwap_ratio_enter_v2,
        )