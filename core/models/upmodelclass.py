import numpy as np
from core.models.up_model_property import Line, Point


class UpModel:
    def __init__(
            self,
            # df,
            t1up,
            t2up,
            t3up,
            t4up,
            t5up,
            first_bar_above_t4up,
            up_take_lines,
            dist_cp_t4_x2,
            HP_up_point,
            LC_break_point,
            point_under_LC,
            LT_break_point,
            CP_up_point,
            dist_cp_t4_x1,
            LT_break_point_close,
            slope_LT,
            intercept_LT,
            # LC_up,
            # slope_LC,
            # intercept_LC,
            # LT_up,
            # slope_LT,
            # intercept_LT,
            # CP_up_point
    ):
        # self.df = df
        self.t1up = t1up
        self.t2up = t2up
        self.t3up = t3up
        self.t4up = t4up
        self.t5up = t5up
        self.first_bar_above_t4up = first_bar_above_t4up
        self.up_take_lines = up_take_lines
        self.dist_cp_t4_x2 = dist_cp_t4_x2
        self.HP_up_point = HP_up_point
        self.LC_break_point = LC_break_point
        self.point_under_LC = point_under_LC
        self.LT_break_point = LT_break_point
        self.CP_up_point = CP_up_point
        self.dist_cp_t4_x1 = dist_cp_t4_x1
        self.LT_break_point_close = LT_break_point_close
        self.slope_LT = slope_LT
        self.intercept_LT = intercept_LT
        # self.LC_up = LC_up
        # self.slope_LC = slope_LC
        # self.intersect_LC = intercept_LC
        # self.LT_up = LT_up
        # self.slope_LT = slope_LT
        # self.intersect_LT = intercept_LT
        # self.CP_up_point = t1up
        # self.dist_cp_t4_x1 = None
        # self.dist_cp_t4_x2 = None
        # self.find_LT_up_breakout_point()
        # self.find_target()
        # self.find_first_bar_above_t4up()
        # self.above_is_faster_breakout()
        # self.find_t5up()
