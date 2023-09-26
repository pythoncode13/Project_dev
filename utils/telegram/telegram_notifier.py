import requests
import config

class TelegramNotifier:
    def __init__(self):
        self.token = config.token
        self.chat_id = config.chat_id
        self.base_url = f'https://api.telegram.org/bot{self.token}/sendMessage'

    def send_message(self, text):
        payload = {'chat_id': self.chat_id, 'text': text}
        requests.post(self.base_url, data=payload)