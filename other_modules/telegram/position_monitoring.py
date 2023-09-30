import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.loader_binance import LoaderBinance
from binance import Client
from core.position_evaluator import PositionEvaluator
from other_modules.telegram.telegram_notifier import TelegramNotifier
from utils.progress_bar_utils import add_progress_bar
import config


class PositionMonitoring:
    def __init__(self):
        self.telegram_notifier = TelegramNotifier()
        self.filepath = config.TELEGRAM_DIR + 'order_in_process_data.csv'

    def notify(self, message_id, s_date_perv, ticker, message):
        self.telegram_notifier.send_message_with_image(
                                                        message_id,
                                                        s_date_perv,
                                                        ticker,
                                                        message
        )

    def check_positions(self):
        # Загрузка существующих данных
        existing_data_df = pd.read_csv(self.filepath, encoding='utf-8-sig')

        # Проходим по каждой строке и проверяем условия
        for index, row in existing_data_df.iterrows():
            s_date_perv = row['Дата']
            ticker = row['Тикер']
            entry = row['Вход']
            stop = float(row['Стоп'])
            take = float(row['Тейк'])
            force_close_date_str = row['Принудительно закрыть']
            message_id = row['message_id']
            percentage_change_take = ((take - entry) / entry) * 100
            percentage_change_stop = ((stop - entry) / entry) * 100

            timeframes = '30MINUTE'
            # Заменяем последние два дефиса на двоеточия
            s_date = ':'.join(s_date_perv.rsplit('-', 2))

            # Получаем текущую дату и время
            current_datetime = datetime.now()
            # Получаем только дату (без времени)
            current_date = current_datetime.date()

            # Плюсуем один день
            u_date = str(current_date + timedelta(days=1))+' 00:00:00'

            # Загружаем DF
            loader = LoaderBinance(
                ticker,
                timeframes,
                s_date,
                u_date,
            )
            loader.get_historical_data()
            df = loader.df

            # Преобразование строки принудительного закрытия в объект datetime
            force_close_date = datetime.strptime(force_close_date_str,
                                                 "%Y-%m-%d %H-%M-%S")

            # Находим индекс первой строки, где 'low' меньше или равно стопу
            stop_point_index = df.loc[df['close'] <= stop].index.min()
            if not isinstance(stop_point_index, (int, np.integer)):
                stop_point_index = None

            # Находим индекс первой строки, где 'high' больше или равно тейку
            take_point_index = df.loc[df['high'] >= take].index.min()
            # print("Тип переменной take_point_index:", type(take_point_index))
            if not isinstance(take_point_index, (int, np.integer)):
                take_point_index = None

            # Создание экземпляра класса PositionEvaluator с нужными параметрами
            evaluator = PositionEvaluator(ticker, current_datetime,
                                          take_point_index,
                                          stop_point_index,
                                          force_close_date,
                                          )

            # Вызов метода evaluate для данного экземпляра
            close_position, close_reason = evaluator.evaluate()

            if close_position:
                # Удаляем строку из файла
                existing_data_df.drop(index, inplace=True)
                if close_reason == 'Take':
                    self.notify(message_id,
                                s_date_perv,
                                ticker,
                                f'#{ticker}\n'
                                f'Тейк достигнут!\n'
                                f'Сделка закрыта {percentage_change_take:.2f}%'
                                )


                if close_reason == 'Stop':
                    self.notify(message_id,
                                s_date_perv,
                                ticker,
                                f'#{ticker}\n'
                                f'Стоп достигнут!\n'
                                f'Сделка закрыта {percentage_change_stop:.2f}%'
                                )
                    # self.send_image(message_id, s_date_perv, ticker)
                if close_reason == 'Force Close':
                    # Последний бар из DataFrame
                    last_bar = df.iloc[-1]
                    # Цена закрытия последнего бара
                    last_close_price = last_bar['close']
                    # Цена входа
                    entry_price = entry
                    # Процентное изменение между ценой входа и ценой закрытия
                    percentage_change = (
                        (last_close_price - entry_price) / entry_price) * 100

                    # Отправляем сообщение с percentage_change
                    self.notify(message_id,
                                s_date_perv,
                                ticker,
                                f'#{ticker}\n'
                                f'Время сделки истекло!\n'
                                f'Сделка закрыта {percentage_change:.2f}%'
                                )

                # Сохранение обновленного файла
                existing_data_df.to_csv(self.filepath, index=False,
                                        encoding='utf-8-sig')
            else:
                # Если сделка не закрыта, печатаем прогресс бар
                client = Client(config.api_key, config.api_secret)
                ticker_price_info = client.get_symbol_ticker(symbol=ticker)
                price = float(ticker_price_info['price'])
                # Рассчитаем процентное изменение
                price_percentage_change = ((price - entry) / entry) * 100
                s_date = datetime.strptime(s_date, '%Y-%m-%d %H:%M:%S')
                add_progress_bar(
                    ticker,
                    s_date,
                    current_datetime,
                    force_close_date,
                    price_percentage_change
                )
        else:
            print("Нет новых сделок.")


def position_monitoring():
    # Вызов функции для проверки позиций
    monitor = PositionMonitoring()
    monitor.check_positions()


# if __name__ == '__main__':
#     config.IMAGES_DIR = 'C:/Users/Home/Desktop/Strategy/MultipointsStrategy_v1 working buy at t4/data/images/'
#     config.TELEGRAM_DIR = 'C:/Users/Home/Desktop/Strategy/MultipointsStrategy_v1 working buy at t4//utils/telegram/'
#     position_monitoring()
