import numpy as np
from core.core import AppInitializer, MultyWorker
import schedule
import time
from datetime import datetime
import config

import warnings

# Игнорировать FutureWarning и RankWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=np.RankWarning)

DEV = True

# Включаем(True)/Выключаем(False) планировщик
ENABLE_SCHEDULER = False


def start():
    """Главный программный цикл."""
    AppInitializer()

    if DEV:
        # tickers = [
        #     'APTUSDT'
        # ]
        tickers = config.tickers
        timeframes = ['30MINUTE']

        s_date = "2023-01-01 00:00:00"
        u_date = "2023-09-30 10:00:00"

        worker = MultyWorker(tickers, timeframes, s_date, u_date,
                             images=False,
                             debug=False,
                             show_progress_bar=True,
                             telegram=False
                             )
        worker.work()

    else:
        tickers = config.tickers
        timeframes = config.timeframes
        s_date = config.s_date
        u_date = config.u_date
        worker = MultyWorker(tickers, timeframes, s_date, u_date,
                             images=False,
                             debug=False,
                             show_progress_bar=True,
                             telegram=True
                             )
        worker.work()


def job():
    """Запускает основную задачу и
    выводит время начала и окончания выполнения."""
    print(f"Job starting at {datetime.now()}")
    start()
    print(f"The job started at {datetime.now()} has completed.")


def print_countdown():
    """
    Выводит обратный отсчет времени до следующего запуска задачи.
    Отображает часы, минуты и секунды до следующего запланированного запуска.
    """
    next_run = min(job.next_run for job in schedule.jobs)
    while datetime.now() < next_run:
        remaining_time = next_run - datetime.now()
        hours, remainder = divmod(remaining_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            f"\rСледующий запуск через: "
            f"{int(hours)}:{int(minutes)}:{int(seconds)}", end=""
        )
        time.sleep(1)
    print("\nЗапуск задачи...")


if __name__ == '__main__':
    if ENABLE_SCHEDULER:
        # Запланируем работу на 0 и 30 минут каждого часа
        print("Программа запущена с автоматическим запуском задач.")
        schedule.every().hour.at(":00").do(job)
        schedule.every().hour.at(":30").do(job)

        while True:
            # Печать обратного отсчета до следующего запуска
            print_countdown()
            schedule.run_pending()
            time.sleep(1)

    else:
        start()  # Если планировщик отключен, тогда запустится функция start
