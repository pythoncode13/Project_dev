from scipy.spatial import distance
from decimal import Decimal
import numpy as np


class HarmonicaParams:
    def __init__(self, model):
        self.model = model

        # Вычисляем точки на LT
        self.point_at_t4 = self.calculate_point_on_LT(model.t4[0], model.LT.slope, model.LT.intercept)
        self.point_at_t2 = self.calculate_point_on_LT(model.t2[0], model.LT.slope, model.LT.intercept)

        # Расчет расстояний и отношений
        self.calculate_distances_and_ratios()

    def calculate_point_on_LT(self, x_value, slope, intercept):
        y_value = slope * Decimal(x_value) + intercept
        return float(x_value), float(y_value)

    def calculate_distances_and_ratios(self):
        # Точки XABCD
        X = (float(self.model.CP[0]), float(self.model.CP[1]))
        A = self.model.t1
        B = self.model.t2
        C = self.model.t3
        D = self.model.t4
        E = self.point_at_t4
        F = self.point_at_t2

        # Вычисляем расстояния
        AB = distance.euclidean(A, B)
        AD = distance.euclidean(A, D)
        AC = distance.euclidean(A, C)
        AE = distance.euclidean(A, E)
        BC = distance.euclidean(B, C)
        BD = distance.euclidean(B, D)

        AF = distance.euclidean(A, F)
        BF = distance.euclidean(B, F)
        FC = distance.euclidean(F, C)
        XF = distance.euclidean(X, F)
        FE = distance.euclidean(F, E)
        FD = distance.euclidean(F, D)

        CD = distance.euclidean(C, D)
        CE = distance.euclidean(C, E)
        DE = distance.euclidean(D, E)

        XE = distance.euclidean(X, E)
        XA = distance.euclidean(X, A)
        XC = distance.euclidean(X, C)
        XB = distance.euclidean(X, B)
        XD = distance.euclidean(X, D)

        # Вычисляем отношения
        # CP
        XE_to_CB = XE / BC
        XA_to_AC = XA / AC
        XA_to_CE = XA / CE
        XE_to_XF = XE / XF
        XD_to_XB = XD / XB
        XB_to_XE = XB / XE
        XB_to_XA = AB / XA

        # t1
        AB_to_BC = AB / BC
        AD_to_AB = AD / AB
        AC_to_BD = AC / BD
        AC_to_CE = AC / CE
        AF_to_FC = AF / FC
        AF_to_CE = AF / CE
        AE_to_FC = AE / FC

        # t2
        BF_to_DF = BF / FD
        BF_to_BC = BF / BC
        AB_to_BF = AB / BF
        BC_to_BD = BC / BD

        # Расчет отношений и сравнение с числами Фибоначчи
        AB_to_XA = AB / XA
        BC_to_AB = BC / AB
        CD_to_BC = CD / BC

        return (XE_to_CB, XA_to_AC, XA_to_CE, XE_to_XF, XD_to_XB, XB_to_XE,
                XB_to_XA, AB_to_BC, AD_to_AB, AC_to_BD, AC_to_CE, AF_to_FC,
                AF_to_CE, AE_to_FC, BF_to_DF, BF_to_BC, AB_to_BF, BC_to_BD,
                AB_to_XA, BC_to_AB, CD_to_BC
                )
