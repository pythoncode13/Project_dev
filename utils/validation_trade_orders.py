from joblib import load
import pandas as pd
import numpy as np
import os
import config


def validation_trade_orders():

    # Загрузка сохраненных моделей
    model_folder = config.UTILS_DIR
    rf_model1 = load(model_folder+'random_forest_classifier_v6.pkl')
    rf_model2 = load(model_folder+'random_forest_mse_best_parameters1.pkl')
    boosting_model = load(model_folder+'бустинг_2_модели.pkl')
    scaler = load(model_folder+'scaler_v1.pkl')

    results_folder = config.RESULTS_DIR
    filename_input = results_folder + '__final_output_new.xlsx'
    filename_output = results_folder + 'predicted_ones_rf_classifier.xlsx'

    # Загрузка новых данных
    new_df = pd.read_excel(filename_input)

    if new_df.empty:
        return None
    # Сохранение первых 7 столбцов в отдельный DataFrame
    first_seven_columns_df = new_df.iloc[:, :8].copy()

    # Удаление первых 7 столбцов
    new_df = new_df.drop(new_df.columns[range(8)], axis=1)

    # Удаление строк с пропущенными значениями
    # new_df = new_df.dropna()

    column_name = new_df.columns[0]
    # print(column_name)

    X_new = new_df.drop(column_name, axis=1)

    # Масштабирование данных для первой модели
    X_new_scaled = scaler.transform(X_new)

    # Получение предсказаний от обеих моделей
    rf1_preds = rf_model1.predict_proba(X_new_scaled)[:, 1].reshape(-1, 1)
    rf2_preds = rf_model2.predict_proba(X_new)[:, 1].reshape(-1, 1)

    # Собираем предсказания в один массив для бустинг-классификатора
    X_boosting = np.hstack([rf1_preds, rf2_preds])

    # # Сделать предсказания с помощью модели бустинга
    # boosting_preds = boosting_model.predict(X_boosting)
    #
    # # Добавьте предсказания в исходный DataFrame
    # new_df['Предсказания'] = boosting_preds
    #
    # # Сохраните результаты в файл Excel
    # output_path = 'предсказания.xlsx'
    # new_df.to_excel(output_path, index=False)

    # Получение вероятностей принадлежности к положительному классу от бустинг-классификатора
    predictions_proba = boosting_model.predict_proba(X_boosting)[:, 1]

    # Задание порога (по умолчанию 0.5)
    threshold = 0.51

    # Применение порога для получения бинарных предсказаний (0 или 1)
    predictions_custom_threshold = (predictions_proba >= threshold).astype(int)

    # # Добавляем предсказания с пользовательским порогом в исходный DataFrame
    # new_df['Предсказания'] = predictions_custom_threshold
    #
    # # Сохраняем результаты в файл Excel
    # output_path = 'предсказания.xlsx'
    # new_df.to_excel(output_path, index=False)

    # Добавляем обратно сохраненные столбцы
    new_df_with_first_columns = pd.merge(first_seven_columns_df, new_df, left_index=True, right_index=True)

    # Фильтрация данных на основе предсказаний с пользовательским порогом (только те строки, где предсказание равно 1)
    trades_to_make = new_df_with_first_columns[predictions_custom_threshold == 1]

    # Сохраняем отфильтрованные данные в файл Excel
    # filename_output = results_folder + 'predicted_ones_rf_classifier.xlsx'
    # Удаляем строки, где столбец "Успешно" не пуст (не равен NaN)
    trades_to_make = trades_to_make[trades_to_make['Успешно'].isna()]
    trades_to_make.to_excel(filename_output, index=False)

    # # Вывод среднего значения столбцов 'Разница' и 'Успешно' (если они есть в вашем DataFrame)
    # if 'Разница' in trades_to_make.columns:
    #     print("Среднее значение 'Разница':", round(trades_to_make['Разница'].mean(), 2))
    # if 'Успешно' in trades_to_make.columns:
    #     print("Среднее значение 'Успешно':", round(trades_to_make['Успешно'].mean(), 2))

    if not trades_to_make.empty:
        # Вывод количества строк в DataFrame
        shape_of_old_df = new_df.shape[0]
        print("\nКоличество строк в DataFrame:", trades_to_make.shape[0], f'({round(trades_to_make.shape[0] * 100/shape_of_old_df, 2)}%)')

        # Получение и вывод параметров модели случайного леса
        # params = boosting_model.get_params()
        # for key, value in params.items():
        #     print(f"{key}: {value}")
        return trades_to_make
    else:
        print("\nКоличество строк в DataFrame: 0")
        return None
