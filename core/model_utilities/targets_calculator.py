
def levels(t1, t2, t3, t4, CP_up_point):
    """ --------------- АКТИВАЦИЯ МОДЕЛИ ------------------ """
    # dist_cp_t4_x2 = t4[0] + (
    #         (t4[0] - float(CP_up_point[0])) * 2)
    #
    # upper_limit = min(int(dist_cp_t4_x2), len(self.df))
    #
    # # Находим бар, который пробил уровень т4
    # first_bar_above_t4 = None
    # t5up = None
    # HP_up_point = None
    # for i in range(int(t4[0]), upper_limit):
    #     if self.df.loc[i, 'high'] > t4[1]:
    #         first_bar_above_t4 = i
    #         break
    #
    # LT_break_point = Point.find_LT_break_point(self.df, t4, upper_limit,
    #                                            slope_LT, intercept_LT)
    #
    # LT_break_point_close = None
    # if LT_break_point:
    #     LT_break_point_close = Point.find_LT_break_point_close(self.df,
    #                                                            t4,
    #                                                            upper_limit,
    #                                                            slope_LT,
    #                                                            intercept_LT)
    #
    # if (
    #         # Если first_bar_above_t4 существует
    #         # и LT_break_point не существует
    #         not LT_break_point and first_bar_above_t4
    #         # ИЛИ
    #         or
    #         # Если first_bar_above_t4 и LT_break_point оба существуют
    #         # и индекс first_bar_above_t4 меньше индекса
    #         # в кортеже LT_break_point
    #         (
    #                 LT_break_point and first_bar_above_t4 and first_bar_above_t4 <
    #                 LT_break_point[0])
    # ):
    #
    #     t5up_index = self.df.loc[(t4[0] + 1):first_bar_above_t4,
    #                  'low'].idxmin()
    #     t5up_price = self.df.loc[t5up_index, 'low']
    #     if t5up_price >= t4[1]:
    #         break
    #     t5up = (t5up_index, t5up_price)
    #
    #     # Проверка пересечение тел свечей т2-т5
    #     t2_candle = self.df.loc[t2[0]]
    #     t5up_candle = self.df.loc[t5up[0]]
    #
    #     t2_upper_body_edge = max(t2_candle['open'],
    #                              t2_candle['close'])
    #     t5up_lower_body_edge = min(t5up_candle['open'],
    #                                t5up_candle['close'])
    #
    #     if t2_upper_body_edge > t5up_lower_body_edge:
    #         continue
    #     else:
    #         # find HP
    #         # проводим линию HP
    #         slope_LT_HP, intercept_LT_HP, LT_up_HP = Line.calculate(
    #             t3, t5up)
    #         # валидация
    #         if Line.check_line(self.df, slope_LT_HP, intercept_LT_HP, t3,
    #                            t5up,
    #                            direction='low'):
    #             t31 = Line.correction_LT_HP(self.df, t3, t5up,
    #                                         slope_LT_HP,
    #                                         intercept_LT_HP)
    #             slope_LT_HP, intercept_LT_HP, LT_up_HP = Line.calculate(
    #                 t31, t5up)
    #         # plt.plot(LT_up_HP[0], LT_up_HP[1], ':', color='purple',
    #         #          linewidth=0.9)
    #
    #         # Поиск точки пересечения прямых LT_up_HP и ЛЦ
    #         # if slope_LT_HP == slope_LC:
    #         #
    #         # x_intersect_LC_LT_up_HP_point = (intercept_LC - intercept_LT_HP) / (
    #         #             slope_LT_HP - slope_LC)
    #         # y_intersect_LC_LT_up_HP_point = slope_LT_HP * x_intersect_LC_LT_up_HP_point + intercept_LT_HP
    #         #
    #         # if y_intersect_LC_LT_up_HP_point > t4[1] + (t4[1] - t1up[1]) * 5:
    #         #     continue
    #         # if x_intersect_LC_LT_up_HP_point < t4[0]:
    #         #     HP_up_point = None
    #         # else:
    #         #     HP_up_point = (x_intersect_LC_LT_up_HP_point, y_intersect_LC_LT_up_HP_point)
    #         HP_up_point = None
    #
    # LC_break_point = None
    # if first_bar_above_t4:
    #     LC_break_point = Point.find_LC_break_point(self.df, t4,
    #                                                dist_cp_t4_x2,
    #                                                slope_LC,
    #                                                intercept_LC)
    #
    # point_under_LC = None
    # if LC_break_point:
    #     point_under_LC = Point.find_LT_break_point(self.df, LC_break_point,
    #                                                dist_cp_t4_x2,
    #                                                slope_LC,
    #                                                intercept_LC)

    """ --------------- УРОВНИ ТЕЙКОВ ------------------ """


    # Вычисляем коэффициенты уравнения прямой
    m = (t2[1] - t1[1]) / (t2[0] - t1[0])
    b = t1[1] - m * t1[0]

    # Расширяем линию тренда на две длины от t1 до t2
    vline_x = t2[0] + 1 * (t2[0] - t1[0])

    # Находим точку пересечения
    x_intersect = vline_x
    y_intersect_up_take = m * x_intersect + b

    up_take_lines = (x_intersect, y_intersect_up_take)

    """ --------------- ------------------ """




