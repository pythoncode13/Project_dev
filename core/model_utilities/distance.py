import numpy as np
from decimal import Decimal, getcontext


class Distance:

    def __init__(self):
        pass

    @staticmethod
    def calculate(point1, point2):
        return point2[0]-point1[0]

    @staticmethod
    def calculate_x(point1, point2, x):
        point1 = float(point1[0])
        point2 = float(point2[0])
        return point2 + ((point2 - point1) * x)

    @staticmethod
    def calculate_distance(point1: np.array, point2: np.array) -> float:
        """
        Вычисляет евклидово расстояние между двумя точками.
        """
        return np.round(np.sqrt(np.sum((point1 - point2) ** 2)), 5)
