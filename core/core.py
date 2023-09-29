import concurrent.futures
import progressbar
import logging
import colorlog

import os
import glob
from multiprocessing import Lock
import pandas as pd

import config
from core.candle_plot import CandleStickPlotter
from core.loader_binance import LoaderBinance

from utils.excel_saver import ExcelSaver
from utils.excel_combiner_autostart import combine_excel_files
from utils.validation_trade_orders import validation_trade_orders
from utils.new_position_to_data import add_position_to_data
from utils.telegram.send_message import TelegramMessage
from utils.telegram.position_monitoring import position_monitoring
from utils.progress_bar_utils import create_progress_bar

from core.trading.trading_one_exp_model_long import prepare_trading_setup, one_exp_model_long
from core.point_combinations.treand_models.price_level_checker import PriceLevelChecker
from core.trading_backtester import StrategySimulator

from core.point_combinations.treand_models.up_trend_model import UpTrendModel
from other_modules.clear_directory import clear_directory


class AppInitializer:
    """Инициализация приложения."""

    def __init__(self):
        progressbar.streams.wrap_stderr()  # Обёртка вокруг stderr для отделения прогресс бара от логов
        colorlog.basicConfig(
            format='%(asctime)s - %(message)s',
            level=config.LOGGING_LEVEL,
        )

        if not os.path.exists(config.RESULTS_DIR):
            os.makedirs(config.RESULTS_DIR)
        if not os.path.exists(config.IMAGES_DIR):
            os.makedirs(config.IMAGES_DIR)
        if not os.path.exists(config.BIGDATA_DIR):
            os.makedirs(config.BIGDATA_DIR)

        # Очищаем папки images и results
        clear_directory(config.IMAGES_DIR)
        clear_directory(config.RESULTS_DIR)


class Worker:
    """Главный обработчик конфигурации."""

    def __init__(self, ticker, timeframe, s_date, u_date, images=False):
        self.ticker = ticker
        self.timeframe = timeframe
        self.s_date = s_date
        self.u_date = u_date
        self.images = images

    def work(self):
        """Выполняем базовый цикл."""
        # Загружаем DF
        loader = LoaderBinance(
            self.ticker,
            self.timeframe,
            self.s_date,
            self.u_date,
        )
        loader.get_historical_data()
        loader.add_indicators()
        df = loader.df

        # Создаем график
        if self.images:
            plot = CandleStickPlotter(df)
            plot.add_candlesticks()

        # Получаем модели-кандидаты для дальнейшего анализа
        candidates_up = UpTrendModel(df).find_candidates()
        # Отправляем модели-кандидаты в модуль торговли
        # Торговля одной модели расширения, лонг
        one_exp_model_long(
                           candidates_up,
                           self.ticker,
                           self.timeframe,
                           self.s_date,
                           self.u_date
                           )

        if self.images:
            s_date_date_only = self.s_date.split()[0]
            u_date_date_only = self.u_date.split()[0]
            filename = (
                config.IMAGES_DIR + f'{self.ticker}-{self.timeframe}-{s_date_date_only}-{u_date_date_only}.png'
            )
            plot.save(filename)


class MultyWorker:
    """Многопоточная обработка конфигурации."""

    def __init__(
        self,
        tickers,
        timeframe_intervals,
        s_date, u_date,
        images=False,
        debug=True,
        show_progress_bar=False,  # Показать прогресс бар
    ):
        self.tickers = tickers
        self.timeframe_intervals = timeframe_intervals
        self.s_date = s_date
        self.u_date = u_date
        self.images = images
        self.debug = debug
        self.lock = Lock()
        self.tasks_completed = 0
        self.show_progress_bar = show_progress_bar

    @property
    def tasks_total(self):
        """Общее количество задач."""
        return len(self.tickers) * len(self.timeframe_intervals)

    def work(self):

        if self.show_progress_bar:  # Проверяем значение свойства
            # Инициализация прогресс бара с кастомными параметрами
            self.progress_bar = create_progress_bar(self.tasks_total)
        logging.info("Начинаю сбор данных и расчет.")

        # Подготавливаем отдельные Worker под каждую конфигурацию
        workers = []
        for ticker in self.tickers:
            for timeframe in self.timeframe_intervals:
                worker = Worker(
                    ticker,
                    timeframe,
                    self.s_date,
                    self.u_date,
                    self.images,
                )
                workers.append(worker)

        if not self.debug:
            # Запускаем многопоточный режим
            with concurrent.futures.ProcessPoolExecutor(
                    max_workers=config.CPU_COUNT,
            ) as executor:
                futures = [executor.submit(worker.work) for worker in workers]
                for future in futures:
                    future.add_done_callback(self.progress_indicator)
        else:
            # Запускаем один поток для отладки
            for worker in workers:
                worker.work()

        # """Обрабатываем полученные файлы."""
        # Соединяем эксель файлы
        combine_excel_files()
        # Выбираем валидные сетапы
        # trades_to_make = validation_trade_orders()

        # Загрузка данных excel
        trades_to_make = pd.read_excel(
            config.RESULTS_DIR + '__final_output_new.xlsx'
        )

        # if new_df.empty:
        #     return None

        if trades_to_make is not None:
            print(trades_to_make)
            # Добавляем уникальные строки в базу
            new_rows = add_position_to_data(trades_to_make)
            # Если среди добавленных строк есть новые,
            # тогда отправляем в Телеграм
            if not new_rows.empty:
                TelegramMessage.send_message_in_telegram(new_rows)
        # Запускаем функцию проверки состояния открытых позиций
        position_monitoring()

    def progress_indicator(self, _):
        with self.lock:
            self.tasks_completed += 1
            if self.show_progress_bar:  # Добавляем эту проверку
                self.progress_bar.update(
                    self.tasks_completed)  # Обновление прогресса
