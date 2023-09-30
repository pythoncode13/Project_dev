import pandas as pd
import config


def add_position_to_data(trades_to_make):
    # Список колонок, которые хотите сохранить
    columns_to_keep = ['Дата', 'ticker', 'Вход', 'Стоп', 'Тейк']

    # Оставляем только нужные колонки в датафрейме
    trades_to_make = trades_to_make.loc[:, columns_to_keep]

    # Округление значений в указанных колонках до 5 знаков после запятой
    trades_to_make['Вход'] = trades_to_make['Вход'].round(5)
    trades_to_make['Стоп'] = trades_to_make['Стоп'].round(5)
    trades_to_make['Тейк'] = trades_to_make['Тейк'].round(5)

    filepath = config.TELEGRAM_DIR + 'data.csv'
    # Загрузка существующих данных из CSV-файла в DataFrame
    existing_data_df = pd.read_csv(filepath, encoding='utf-8-sig')

    # Объединение существующих данных и новых данных
    combined_data = pd.concat([existing_data_df, trades_to_make]).drop_duplicates()

    # Сортировка DataFrame по столбцу "Дата"
    combined_data = combined_data.sort_values(by=['Дата'])

    # Сохранение объединенного датафрейма в CSV-файл
    combined_data.to_csv(filepath, index=False, encoding='utf-8-sig')

    # Находим строки, которые присутствуют только в combined_data,
    # но не в existing_data_df
    new_rows = combined_data.merge(existing_data_df, how='outer',
                                   indicator=True)
    new_rows = new_rows[new_rows['_merge'] == 'left_only'].drop(
        columns=['_merge'])

    return new_rows
