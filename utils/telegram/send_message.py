import requests
from datetime import datetime, timedelta
import pandas as pd
import os

import config

config.IMAGES_DIR = 'C:/Users/Home/Desktop/Strategy/MultipointsStrategy_v1 working buy at t4/data/images/'
config.TELEGRAM_DIR = 'C:/Users/Home/Desktop/Strategy/MultipointsStrategy_v1 working buy at t4//utils/telegram/'


def get_message_id_by_date(date, filepath):
    existing_data_df = pd.read_csv(filepath, encoding='utf-8-sig')
    matching_row = existing_data_df[existing_data_df['Дата'] == date]
    if matching_row.empty:
        return None
    return matching_row['message_id'].iloc[0]


def add_order_in_process(row, message_id):

    filepath = config.TELEGRAM_DIR + 'order_in_process_data.csv'
    # Загрузка существующих данных
    existing_data_df = pd.read_csv(filepath, encoding='utf-8-sig')

    # Проверка на дубликаты
    if existing_data_df.equals(row):
        print("Дубликат найден, новая строка не добавлена.")
        return

    # Добавление новой строки
    new_data_df = pd.DataFrame([row], columns=existing_data_df.columns)
    new_data_df['message_id'] = message_id  # Добавляем ID сообщения в новую строку
    combined_data = pd.concat([existing_data_df, new_data_df])

    # Сохранение в CSV
    combined_data.to_csv(filepath, index=False, encoding='utf-8-sig')


def create_message(row, balance=2500, risk=0.01):
    entry_price = row['Вход']
    stop_price = row['Стоп']
    entry_amount = balance * risk * (entry_price / (entry_price - stop_price))

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


def send_message_in_telegram():

    # '''__________________________________________________________________'''
    folder = os.path.join(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    filename = 'data — копия.csv'
    filepath = os.path.join(folder, filename)


    print('test')
    # Загрузка данных
    new_rows = pd.read_csv(filepath, encoding='utf-8-sig')
    # '''__________________________________________________________________'''

    # Токен бота
    TOKEN = "6048284555:AAHP2qXwhOEpVSiMOWtLGJ_eUlYVirpnDVc"

    # Идентификатор чата
    chat_id = "@bot_torgyet"

    # URL для отправки сообщения
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    # Проходим по каждой строке и отправляем сообщение
    for index, row in new_rows.iterrows():
        text_message, message_data = create_message(row, balance=1600, risk=0.01)

        # Параметры для отправки сообщения
        payload = {
            "chat_id": chat_id,
            "text": text_message,
            "parse_mode": "Markdown"  # Указываем, что используем разметку Markdown
        }

        # Отправка POST-запроса
        response = requests.post(url, data=payload)



        # Проверка успешности запроса
        if response.status_code == 200:
            print(f"Сообщение для {row['ticker']} отправлено!")
            # Добавляем информацию о позиции в файл order_in_process_data.csv
            # add_order_in_process((message_data))
            # Получаем ID отправленного сообщения
            message_id = response.json().get('result', {}).get('message_id')

            # Добавляем строку с ID сообщения в файл
            add_order_in_process(message_data, message_id)
            # Отправляем изображение

        else:
            print(f"Ошибка отправки сообщения для {row['ticker']}: {response.content}")

        # Отправка изображения
        image_path = f"{config.IMAGES_DIR}{row['ticker']}-{row['Дата']}.png"
        with open(image_path, 'rb') as image:
            payload_image = {
                "chat_id": chat_id,
                "caption": f"#{row['ticker']}",
                "parse_mode": "Markdown",
                "reply_to_message_id": message_id
            }
            files = {'photo': image}
            url_image = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
            response_image = requests.post(url_image, data=payload_image,
                                           files=files)
            # Проверка успешности запроса
        if response_image.status_code == 200:
            print(f"Изображение для {row['ticker']} отправлено!")
        else:
            print(f"Ошибка отправки изображения для {row['ticker']}: {response.content}")


if __name__ == '__main__':
    send_message_in_telegram()
