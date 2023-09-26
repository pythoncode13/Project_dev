class Distance:

    def __init__(self):
        pass

    @staticmethod
    def calculate(point1, point2):
        return point2[0]-point1[0]

    @staticmethod
    def calculate_x1(point1, point2):
        point1 = float(point1[0])
        point2 = float(point2[0])
        return point2 + ((point2 - point1) * 1)

    @staticmethod
    def calculate_x2(point1, point2):
        point1 = float(point1[0])
        point2 = float(point2[0])
        return point2 + ((point2 - point1) * 2)

    @staticmethod
    def calculate_x(point1, point2, x):
        point1 = float(point1[0])
        point2 = float(point2[0])
        return point2 + ((point2 - point1) * x)
