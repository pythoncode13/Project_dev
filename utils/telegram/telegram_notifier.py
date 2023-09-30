import requests
import json
import config


class TelegramNotifier:
    def __init__(self):
        self.token = config.token
        self.chat_id = config.chat_id
        self.url_base = config.url_base
        self.url_message = config.url_message
        self.url_image = config.url_image

    def send_message(self, message_id, text):
        payload = {
            'chat_id': self.chat_id,
            'text': text,
            "reply_to_message_id": message_id,
        }
        requests.post(self.url_message, data=payload)

    def send_image(self, message_id, date, ticker):
        # Отправка изображения
        image_path = f"{config.IMAGES_DIR}{ticker}-{date}.png"
        with open(image_path, 'rb') as image:
            payload_image = {
                "chat_id": self.chat_id,
                "caption": f"#{ticker}",
                "parse_mode": "Markdown",
                "reply_to_message_id": message_id
            }
            files = {'photo': image}
            response_image = requests.post(self.url_image, data=payload_image,
                                           files=files)
            # Проверка успешности запроса
        if response_image.status_code == 200:
            print(f"Изображение для {ticker} отправлено!")
        else:
            print(f"Ошибка отправки изображения для {ticker}: {response_image.content}")

    def send_combined_message(self, text, image_path):
        # Шаг 1: Загрузим изображение и получим его file_id
        with open(image_path, 'rb') as image:
            files = {'photo': image}
            payload = {"chat_id": self.chat_id}
            response = requests.post(f"{self.url_base}sendPhoto", data=payload,
                                     files=files)
            if response.status_code == 200:
                file_id = json.loads(response.content)['result']['photo'][-1][
                    'file_id']
            else:
                print("Не могу загрузить фото")
                return

        # Шаг 2: Отправляем медиа-группу
        media_group = [
            {"type": "photo", "media": file_id},
            {"type": "text", "caption": text}
        ]

        payload = {
            "chat_id": self.chat_id,
            "media": json.dumps(media_group)
        }

        response = requests.post(f"{self.url_base}sendMediaGroup",
                                 data=payload)

        # Проверка успешности запроса
        if response.status_code == 200:
            print(f"Сообщение и изображение успешно отправлены!")
        else:
            print(f"Ошибка: {response.content}")
