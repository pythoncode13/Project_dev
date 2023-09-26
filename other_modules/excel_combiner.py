"""Комбинирование Excel файлов из results."""

import pandas as pd
import os
from openpyxl import load_workbook

from utils.excel_saver import ExcelSaver

all_dfs = []
folder = '../data/results/'

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
# Создаем новый файл Excel и сохраняем итоговый df в него
# filename = '../data/results/__final_output_new.xlsx'
filename = '__final_output_new.xlsx'
filepath = folder+filename
with pd.ExcelWriter(filepath, engine='openpyxl') as wr:
    final_df.to_excel(wr, index=False)

# Выравнивание столбцов
wb = load_workbook(filepath)
ws = wb.active
ExcelSaver.adjust_column_width(ws)
wb.save(filepath)
