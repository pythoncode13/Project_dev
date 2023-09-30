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

    def send_message_with_image(self, message_id, date, ticker, text):
        # Отправка изображения
        image_path = f"{config.IMAGES_DIR}{ticker}-{date}.png"
        with open(image_path, 'rb') as image:
            payload_image = {
                "chat_id": self.chat_id,
                "caption": text,
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


