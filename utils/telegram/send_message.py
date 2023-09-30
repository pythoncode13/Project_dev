import pandas as pd
import requests
from datetime import datetime, timedelta

import config


class TelegramMessage:
    def __init__(self):
        # Токен бота
        self.token = config.token
        # Идентификатор чата
        self.chat_id = config.chat_id
        # URL для отправки сообщения
        self.url_message = config.url_message
        self.url_image = config.url_image

    @staticmethod
    def get_message_id_by_date(date, filepath):
        """Получение ID сообщения по дате сделки."""
        existing_data_df = pd.read_csv(filepath, encoding='utf-8-sig')
        matching_row = existing_data_df[existing_data_df['Дата'] == date]
        if matching_row.empty:
            return None
        return matching_row['message_id'].iloc[0]

    @staticmethod
    def add_order_in_process(row, message_id):
        """Добавление обработанной строки в файл с активными сделками."""
        filepath = config.TELEGRAM_DIR + 'order_in_process_data.csv'
        # Загрузка существующих данных
        existing_data_df = pd.read_csv(filepath, encoding='utf-8-sig')

        # Проверка на дубликаты
        if existing_data_df.equals(row):
            print("Дубликат найден, новая строка не добавлена.")
            return

        # Добавление новой строки
        new_data_df = pd.DataFrame([row], columns=existing_data_df.columns)
        new_data_df[
            'message_id'] = message_id  # Добавляем ID сообщения в новую строку
        combined_data = pd.concat([existing_data_df, new_data_df])

        # Сохранение в CSV
        combined_data.to_csv(filepath, index=False, encoding='utf-8-sig')

    @staticmethod
    def create_message(row, balance=2500, risk=0.01):
        """Создание сообщения о новой сделке."""
        entry_price = row['Вход']
        stop_price = row['Стоп']
        entry_amount = balance * risk * (
                    entry_price / (entry_price - stop_price))

        # Преобразование строки с датой в объект datetime
        date_format = "%Y-%m-%d %H-%M-%S"
        entry_date = datetime.strptime(row['Дата'], date_format)

        # Добавление 300 минут к дате
        force_close_date = entry_date + timedelta(minutes=3000)

        # Преобразование объекта datetime обратно в строку
        force_close_date_str = force_close_date.strftime(date_format)

        message_data = {
            "Дата": row['Дата'],
            "Позиция": "LONG",
            "Тикер": row['ticker'],
            "Вход": entry_price,
            "Стоп": stop_price,
            "Тейк": row['Тейк'],
            "Сумма входа": f"{entry_amount:.2f}",
            "Принудительно закрыть": force_close_date_str
        }

        text_message = (
            f"Дата: {message_data['Дата']}\n"
            f"Позиция: {message_data['Позиция']}\n"
            f"Тикер: #**{message_data['Тикер']}**\n"
            f"Вход: {message_data['Вход']}\n"
            f"Стоп: {message_data['Стоп']}\n"
            f"Тейк: {message_data['Тейк']}\n"
            f"Сумма входа: {message_data['Сумма входа']}\n"
            f"Принудительно закрыть: {message_data['Принудительно закрыть']}"
        )

        return text_message, message_data

    def send_message(self, row, text_message, message_data):
        date = row['Дата']
        ticker = row['ticker']

        # Отправка изображения
        image_path = f"{config.IMAGES_DIR}{ticker}-{date}.png"
        with open(image_path, 'rb') as image:
            payload_image = {
                "chat_id": self.chat_id,
                "caption": text_message,
                "parse_mode": "Markdown",
            }
            files = {'photo': image}
            response = requests.post(self.url_image, data=payload_image,
                                     files=files)
            # Проверка успешности запроса
        if response.status_code == 200:
            print(f"Изображение для {ticker} отправлено!")
            # Получаем ID отправленного сообщения
            message_id = response.json().get('result', {}).get('message_id')

            # Добавляем информацию о позиции и ID сообщения
            # в файл order_in_process_data.csv
            TelegramMessage.add_order_in_process(message_data, message_id)
        else:
            print(
                f"Ошибка отправки изображения для {ticker}: {response.content}"
            )

    def send_message_in_telegram(self, new_rows):
        """
        Получает новые строки, создает сообщение, отправляет сообщение и
        изображение, добавляет строки(сделки) в файл для наблюдения -
        order_in_process_data.csv
        """
        # Проходим по каждой строке и отправляем сообщение
        for index, row in new_rows.iterrows():
            # Создаем сообщение
            text_message, message_data = TelegramMessage.create_message(
                row, balance=1600, risk=0.01
            )

            # Отправляем сообщение
            self.send_message(row, text_message, message_data)


# if __name__ == '__main__':
#     config.IMAGES_DIR = 'C:/Users/Home/Desktop/Strategy/ \
#     MultipointsStrategy_v1 working buy at t4/data/images/'
#     config.TELEGRAM_DIR = 'C:/Users/Home/Desktop/Strategy/ \
#     MultipointsStrategy_v1 working buy at t4//utils/telegram/'
#     # '''_________________________________________________________________'''
#     filepath = config.TELEGRAM_DIR + 'data — копия.csv'
#
#     print('test')
#     # Загрузка данных
#     new_rows = pd.read_csv(filepath, encoding='utf-8-sig')
#     # '''_________________________________________________________________'''
#     TelegramMessage().send_message_in_telegram(new_rows)
