import pandas as pd
from core.position_evaluator import PositionEvaluator
from core.for_since.calculate_params import extract_features


class StrategySimulator_3points:
    def __init__(self):
        pass

    def trade_process(self, df, activated_models, ticker):
        """Получает входные данные сетапа,
        симулирует торговлю."""
        all_other_parameters_up = []

        for parameters in activated_models:
            entry_index = parameters.variable
            entry_date = parameters.entry_date
            # date_format = '%Y-%m-%d %H:%M:%S'  # Пример формата даты
            # entry_date = entry_date_not_str.strftime(date_format)
            entry_price = parameters.t2[1]
            take_price = parameters.up_take_lines[1]
            stop_price = parameters.t1[1]

            df = df.copy()  # Создаем копию, не изменяем исходный DataFrame
            df['dateTime'] = pd.to_datetime(df['dateTime'])  # Преобразуем дату
            # Обрезаем датафрейм по интересующему нас диапазону
            close_index = entry_index + 101  # конечный индекс
            upper_limit = min(close_index, len(df))
            sub_df = df.iloc[entry_index:upper_limit]
            # Создаем экземпляр класса PositionEvaluator
            # Отправляем ему параметры на проторговку.
            evaluator = PositionEvaluator(sub_df,
                                          ticker,
                                          entry_date,
                                          take_price,
                                          stop_price,
                                          force_close_minutes=3000
                                          )

            # Вызов метода evaluate, получаем результат
            close_position, close_reason, close_point = evaluator.evaluate()

            # Если позиция закрыта,
            # тогда определяем параметры проведенной сделки
            if close_position:
                # Определяем переменную с ценой закрытия сделки
                close_position_price = close_point[1]
                # Считаем изменение цены в % между ценой входа
                # и закрытия сделки
                diff = (
                        (close_position_price - entry_price)
                        / entry_price * 100
                )
                # Определяем была ли сделка успешной или нет
                profit_or_lose = 1 if diff > 0 else 0
                # Считаем длительность сделки
                open_to_close_trade_duration = (
                        close_point[0] - entry_index
                )
            else:
                close_position_price = None
                diff = None
                profit_or_lose = None
                open_to_close_trade_duration = None

            stop_percent_difference = 100 - stop_price * 100 / entry_price

            take_percent_difference = take_price * 100 / entry_price - 100

            features = extract_features(df,
                                       parameters.t1,
                                       parameters.t2,
                                       parameters.t3)

            # Формируем все в одну переменную, которая содержит кортеж с
            # параметрами сделки.
            result_trade = {
                # 'entry_date': entry_date_not_str,
                'ticker': ticker,
                'open_to_close_trade_duration': open_to_close_trade_duration,
                'entry_price': entry_price,
                'stop_price': stop_price,
                'take_price': take_price,
                'close_position_price': close_position_price,
                'diff': diff,
                'profit_or_lose': profit_or_lose,
                'stop_percent_difference': stop_percent_difference,
                'take_percent_difference': take_percent_difference,
            }

            result_trade.update(features)


            all_other_parameters_up.append(result_trade)
        # Преобразование словаря в DataFrame
        df = pd.DataFrame(all_other_parameters_up)


        # Сохранение DataFrame в Excel
        file_path = 'output.xlsx'
        df.to_excel(file_path, index=False)
        return all_other_parameters_up
