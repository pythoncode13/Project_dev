import progressbar

def create_progress_bar(tasks_total):
    return progressbar.ProgressBar(
        max_value=tasks_total,
        widgets=[
            '\033[32mОбработка: ',
            progressbar.Percentage(),
            ' ',
            progressbar.Bar(marker='▓', left='[', right=']', fill='░', fill_left=True),
            ' ',
            '\033[32mВремя работы: ',
            progressbar.Timer(format='%(elapsed)s'),
            '\033[0m',
        ]
    )


def create_static_progress_bar(percentage):
    bar_length = 30
    filled_length = int(round(bar_length * percentage / 100))
    bar = '●' * filled_length + '○' * (bar_length - filled_length)
    # return f'\u001b[38;5;65m{percentage:.1f}% [{bar}]\u001b[0m'
    return f'\u001b[38;5;65m[{bar}]\u001b[0m'


def add_progress_bar(ticker, s_date, current_datetime, force_close_date, price_percentage_change):
    """Добавляем прогресс бар."""
    total_minutes = int((force_close_date - s_date).total_seconds() / 60)
    elapsed_minutes = int((current_datetime - s_date).total_seconds() / 60)
    elapsed_minutes = min(elapsed_minutes, total_minutes)
    percentage_complete = (elapsed_minutes / total_minutes) * 100
    remaining_time = force_close_date - current_datetime
    remaining_time_formatted = str(remaining_time).split('.')[0]  # Форматирование для убирания микросекунд

    # Получение строки прогресс-бара
    progress_bar_string = create_static_progress_bar(percentage_complete)

    print(
        f"\u001b[38;5;107mСделка: {ticker}\u001b[0m {progress_bar_string} "
        f"{price_percentage_change:.1f}%\n"
        f"\u001b[38;5;107mДо закрытия: {remaining_time_formatted}\u001b[0m\n")
