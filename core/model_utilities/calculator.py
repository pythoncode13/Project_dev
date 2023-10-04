class Calculate:
    def __init__(self):
        pass

    @staticmethod
    def percentage_change(original, new):
        """Рассчитывает процентное изменение числа."""
        if original == 0:
            return "На ноль делить нельзя!"

        percentage_change = ((new - original) / original) * 100
        return percentage_change

    @staticmethod
    def percentage_change_short(original, new):
        """Рассчитывает процентное изменение числа."""
        if original == 0:
            return "На ноль делить нельзя!"

        percentage_change = ((original - new) / original) * 100
        return percentage_change
