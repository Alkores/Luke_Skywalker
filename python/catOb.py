# Импорт библиотек
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Загрузка данных из CSV
data = pd.read_csv(r"D:\Download\train_dataset_train_data_NPF\train_data\train1_data.csv", encoding='utf-8', sep=';')  # Входной файл с данными для обучения

# Подготовка данных
feature_columns = ['gndr', 'brth_yr', 'prsnt_age', 'accnt_status', 'pnsn_age', 
                   'avrg_sum', 'sum_sum', 'min_sum', 'max_sum', 'lrgst_sum', 
                   'mnmm_sum', 'one_for_trnsctn', 'trnsctns_frnqnc', 
                   'trnsctns_frnqnc_max', 'trnsctns_frnqnc_min', 
                   'frnqnt_trnsctn', 'frnqnt_trnsctn_1', 'frnqnt_trnsctn_0', 
                   'nmbr_of_trnsltns', 'trnsltn_vltn']
X = data[feature_columns]  # Признаки
y = data['erly_pnsn_flg']  # Целевая переменная

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, eval_metric='Accuracy', verbose=100)
model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

# Сохранение модели
model.save_model(r"D:\Download\train_dataset_train_data_NPF\train_data\catboost_model.cbm")  # Сохранение модели в файл

# Оценка модели на тестовых данных
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Точность модели на тестовых данных: {accuracy:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("Матрица путаницы:\n", conf_matrix)

# Кросс-валидация
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Средняя точность при кросс-валидации: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Прогноз для новых данных (входной файл с эталоном)
new_data = pd.read_csv(r"D:\Download\train_dataset_train_data_NPF\train_data\predict1_data.csv", encoding='utf-8', sep=';')  # Входной файл с данными для предсказания

# Проверка наличия необходимых столбцов в новых данных
if all(col in new_data.columns for col in feature_columns):
    new_data_predictions = model.predict(new_data[feature_columns])  # Прогноз

    # Если в new_data есть истинные значения, оценка модели на новых данных
    if 'erly_pnsn_flg' in new_data.columns:
        y_true = new_data['erly_pnsn_flg']  # Истинные значения
        new_accuracy = accuracy_score(y_true, new_data_predictions)
        new_f1 = f1_score(y_true, new_data_predictions)
        new_precision = precision_score(y_true, new_data_predictions)
        new_recall = recall_score(y_true, new_data_predictions)
        new_conf_matrix = confusion_matrix(y_true, new_data_predictions)

        print(f"Точность модели на новых данных: {new_accuracy:.2f}")
        print(f"F1-Score на новых данных: {new_f1:.2f}")
        print(f"Precision на новых данных: {new_precision:.2f}")
        print(f"Recall на новых данных: {new_recall:.2f}")
        print("Матрица путаницы на новых данных:\n", new_conf_matrix)

        # Сохранение предсказаний в CSV
        output = new_data.copy()
        output['predicted_erly_pnsn_flg'] = new_data_predictions  # Добавление предсказаний в выходной файл
        output.to_csv(r"D:\Download\train_dataset_train_data_NPF\train_data\VALID_data.csv", index=False)  # Выходной файл с предсказаниями
    else:
        print("В файле предсказаний отсутствует столбец 'erly_pnsn_flg' для сверки.")
else:
    print("В файле новых данных отсутствуют необходимые столбцы для предсказания.")
