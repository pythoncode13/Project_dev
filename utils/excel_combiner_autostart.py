"""Комбинирование Excel файлов из results."""

import pandas as pd
import os
from openpyxl import load_workbook

import config
from utils.excel_saver import ExcelSaver


def combine_excel_files():

    all_dfs = []
    folder = config.RESULTS_DIR

    for file in os.listdir(folder):
        if file.endswith('.xlsx') and (file.startswith('long') or file.startswith('short')):
            df = pd.read_excel(os.path.join(folder, file))

            # Округляем значения в столбце до 10 знаков после запятой
            df['Разница'] = df['Разница'].round(5)

            # Удаляем дубликаты в каждом df
            df = df.drop_duplicates(subset=['Разница'])

            # Добавляем этот df в список
            all_dfs.append(df)

    # Объединяем все df в один
    final_df = pd.concat(all_dfs, ignore_index=True)

    # # Удаляем строки, где столбец "Успешно" не пуст (не равен NaN)
    # final_df = final_df[final_df['Успешно'].isna()]

    # Удаляем дубликаты в df по совпадению в столбцах "Дата", "ticker", "Вход"
    final_df = final_df.drop_duplicates(subset=['Дата', 'ticker', 'Вход'])

    # Сортировка DataFrame по столбцу "Дата"
    final_df = final_df.sort_values(by=['Дата'])

    # Создаем новый файл Excel и сохраняем итоговый df в него
    filename = '__final_output_new.xlsx'
    filepath = os.path.join(folder, filename)
    with pd.ExcelWriter(filepath, engine='openpyxl') as wr:
        final_df.to_excel(wr, index=False)

    # Выравнивание столбцов
    wb = load_workbook(filepath)
    ws = wb.active
    ExcelSaver.adjust_column_width(ws)
    wb.save(filepath)
