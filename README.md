# Luke_Skywalker
Проект представлен командой Luke_Skywalker для окружного хакатона ПФО.
## Команда 
- **Глеб Замалетдинов** - Капитан команды :crown:
- **Дима Елисеев** - Дизайнер :art:
- **Святослав Бельковский** - Инженер программист :car:
- **Сергей Захаров** - Инженер программист :moyai:
```
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣶⣄⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣦⣄⣀⡀⣠⣾⡇⠀⠀⠀⠀
⠀⠀💗⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀
⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⢿⣿⣿⡇⠀⠀⠀⠀
⠀⣶⣿⣦⣜⣿⣿⣿⡟⠻⣿⣿⣿⣿⣿⣿⣿⡿⢿⡏⣴⣺⣦⣙⣿⣷⣄⠀⠀⠀
⠀⣯⡇⣻⣿⣿⣿⣿⣷⣾⣿⣬⣥⣭⣽⣿⣿⣧⣼⡇⣯⣇⣹⣿⣿⣿⣿⣧⠀⠀
⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠸⣿⣿⣿⣿⣿⣿⣿⣷
```
## Оглавление :eyes:
1. [Описание проекта](#описание-проекта)
2. [Цель проекта](#цель-проекта)
3. [Задача](#задача)
4. [Стек технологий](#стек-технологий)
5. [Подготовка данных](#Подготовка-данных-обработка-клиентов-и-транзакций)
6. [Построение и оценка модели](#Построение-и-оценка-модели-предсказания-раннего-выхода-на-пенсию)

## Кейс :briefcase:
Прогнозирование раннего выхода на пенсию.

### Кейсодержатель
АО «Негосударственный пенсионный фонд «БУДУЩЕЕ»
## Описание проекта
### Проблема :interrobang:
Современные государственные пенсионные системы сталкиваются с вызовами, обусловленными изменением демографической структуры. Увеличение продолжительности жизни и снижение рождаемости увеличивают финансовую нагрузку, так как государственные пенсионные системы должны выплачивать пенсии дольше. С другой стороны, для негосударственных пенсионных фондов (НПФ) ранний выход на пенсию клиентов уменьшает объем взносов и чистую прибыль, поскольку сокращает время для инвестиционной активности.

### Цель проекта :black_nib:
Создание модели машинного обучения, которая оценит вероятность досрочного выхода на пенсию клиента ОПС или НПФ. Модель позволит более точно прогнозировать денежные потоки и финансовые обязательства для оптимального планирования.

 ## Задача :bulb:
Задача проекта — построить модель, способную предсказывать, обратится ли клиент за досрочной пенсией, основываясь на его характеристиках и макроэкономических показателях. Это знание позволяет государственным и негосударственным пенсионным фондам:

- учитывать риски и планировать финансовые обязательства,
- адаптировать инвестиционные стратегии,
- укреплять устойчивость пенсионной системы.

### Стек технологий :computer:


  <img src= "https://i.imgur.com/Y4JublQ.jpg" title="Jupyter Notebook" width="120" height="100"/>&nbsp; <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original-wordmark.svg" title="Python" width="100" height="100"/>&nbsp; <img src="https://i.imgur.com/DZ0ilNv.png?1" title="CatBoost" width="120" height="100"/>&nbsp;


## Подготовка данных обработка клиентов и транзакций :scissors:
 
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Укажите пути к исходным файлам
clients_file_path = r"D:\Download\train_dataset_train_data_NPF\train_data\cntrbtrs_clnts_ops_trn.csv"
transactions_file_path = r"D:\Download\train_dataset_train_data_NPF\train_data\trnsctns_ops_trn.csv"
# Укажите путь для сохранения новой таблицы
predict_data_file_path = r"D:\Download\train_dataset_train_data_NPF\train_data\train1_data.csv"
train_data_file_path = r"D:\Download\train_dataset_train_data_NPF\train_data\predict1_data.csv"

# Чтение клиентов с оптимизацией типов данных
clients_df = pd.read_csv(clients_file_path, sep=';', encoding='ANSI', dtype={
    'gndr': 'category',
    'accnt_status': 'category',
    'brth_yr': 'int16',
    'prsnt_age': 'int8',
    'pnsn_age': 'int8'
})

# Кодирование пола: 'м' = 1, 'ж' = 0
clients_df['gndr'] = clients_df['gndr'].map({'м': 1, 'ж': 0}).astype('int8')

# Кодирование статуса счета: 'Накопительный период' = 1, 'Выплатной период' = 0
clients_df['accnt_status'] = clients_df['accnt_status'].map({'Накопительный период': 1, 'Выплатной период': 0}).astype('int8')

# Создаем пустой DataFrame для хранения обработанных данных транзакций
transactions_processed = []

# Чтение и обработка транзакций по частям
for chunk in pd.read_csv(transactions_file_path, sep=';', encoding='utf-8', chunksize=100000, dtype={
    'mvmnt_type': 'int8',
    'sum': 'float32'
}):
    # Преобразование дат
    chunk['oprtn_date'] = pd.to_datetime(chunk['oprtn_date'], errors='coerce')

    # Удаление строк с NaT в 'oprtn_date'
    chunk = chunk.dropna(subset=['oprtn_date'])

    # Добавление обработанного чанка в список
    transactions_processed.append(chunk)

# Конкатенация обработанных частей в один DataFrame
transactions_df = pd.concat(transactions_processed, ignore_index=True)

# Уменьшаем использование памяти, преобразуя 'oprtn_date' в int
transactions_df['oprtn_date'] = transactions_df['oprtn_date'].astype('int64')

# Объединение таблиц по 'accnt_id' с использованием метода 'outer'
merged_df = pd.merge(clients_df, transactions_df, on='accnt_id', how='left')

# Вычисление количества переводов для каждого клиента
nmbr_of_trnsltns = merged_df.groupby('clnt_id').size().reset_index(name='nmbr_of_trnsltns')

# Создаем новую колонку для оценки
merged_df['transaction_value'] = merged_df['sum'] * np.where(merged_df['mvmnt_type'] == 1, 1, -1)

# Вычисляем общую положительность переводов
positive_sum = merged_df[merged_df['transaction_value'] > 0].groupby('clnt_id')['transaction_value'].sum().reset_index(name='positive_sum')
negative_sum = merged_df[merged_df['transaction_value'] < 0].groupby('clnt_id')['transaction_value'].sum().reset_index(name='negative_sum')

# Объединяем положительные и отрицательные суммы
trnsltn_vltn = pd.merge(positive_sum, negative_sum, on='clnt_id', how='outer').fillna(0)

# Рассчитываем trnsltn_vltn как нормализованное значение от 0 до 1
trnsltn_vltn['trnsltn_vltn'] = trnsltn_vltn.apply(
    lambda x: round((x['positive_sum'] / (x['positive_sum'] - x['negative_sum'] + 1e-10)), 2) if (x['positive_sum'] + abs(x['negative_sum'])) > 0 else 0, axis=1
)


# Вычисление необходимых столбцов
def calculate_aggregations(group):
    transaction_value = group['transaction_value']
    oprtn_date = group['oprtn_date'].sort_values()
    mvmnt_type = group['mvmnt_type']
    
    result = {
        'avrg_sum': transaction_value.mean(),
        'sum_sum': transaction_value.sum(),
        'min_sum': transaction_value[transaction_value < 0].sum(),
        'max_sum': transaction_value[transaction_value > 0].sum(),
        'lrgst_sum': transaction_value.max(),
        'mnmm_sum': transaction_value.min(),
        'one_for_trnsctn': transaction_value.sum() / len(transaction_value) if len(transaction_value) > 0 else 0,
        'trnsctns_frnqnc': (oprtn_date.diff().mean() / 1e9 if len(oprtn_date) > 1 else np.nan),  # Преобразуем в дни
        'trnsctns_frnqnc_max': (oprtn_date.diff().max() / 1e9 if len(oprtn_date) > 1 else np.nan),
        'trnsctns_frnqnc_min': (oprtn_date.diff().min() / 1e9 if len(oprtn_date) > 1 else np.nan),
        'frnqnt_trnsctn': mvmnt_type.mode()[0] if not mvmnt_type.mode().empty else 0,
        'frnqnt_trnsctn_1': (mvmnt_type == 1).sum(),
        'frnqnt_trnsctn_0': (mvmnt_type == 0).sum(),
    }
    return pd.Series(result)

tqdm.pandas(desc="Обработка групп")
aggregated_data = merged_df.groupby('clnt_id').progress_apply(calculate_aggregations).reset_index()

# Объединяем данные по количеству переводов и положительности переводов
aggregated_data = pd.merge(aggregated_data, nmbr_of_trnsltns, on='clnt_id', how='outer')
aggregated_data = pd.merge(aggregated_data, trnsltn_vltn[['clnt_id', 'trnsltn_vltn']], on='clnt_id', how='outer')

# Объединяем с клиентами
final_df = pd.merge(clients_df[['clnt_id', 'gndr', 'brth_yr', 'prsnt_age', 'erly_pnsn_flg', 'accnt_status', 'pnsn_age']],
                    aggregated_data, on='clnt_id', how='left')

# Удаляем строки, где 'trnsltn_vltn' является NaN
final_df.dropna(subset=['trnsltn_vltn'], inplace=True)

# Удаляем строки с нулевыми значениями в 'trnsltn_vltn' после формирования таблицы
final_df = final_df[final_df['trnsltn_vltn'] != 0]

# Разделение данных на обучающую и тестовую выборки (70% на 30%)
train_df, predict_df = train_test_split(final_df, test_size=0.3, random_state=42)

# Удаляем столбец clnt_id у предикта и тренировки
train_df.drop(columns=['clnt_id'], inplace=True)
predict_df.drop(columns=['clnt_id'], inplace=True)

# Балансировка erly_pnsn_flg в тренировочной выборке
count_1 = train_df['erly_pnsn_flg'].value_counts().get(1, 0)
train_df_0 = train_df[train_df['erly_pnsn_flg'] == 0].sample(n=count_1, random_state=42)
train_df_1 = train_df[train_df['erly_pnsn_flg'] == 1]
train_df_balanced = pd.concat([train_df_0, train_df_1])

# Перемешиваем сбалансированную выборку
train_df_balanced = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Сохранение новых таблиц в CSV
train_df_balanced.to_csv(train_data_file_path, index=False, encoding='utf-8', sep=';')
predict_df.to_csv(predict_data_file_path, index=False, encoding='utf-8', sep=';')

print("Новые таблицы успешно созданы и сохранены по пути:")
print(train_data_file_path)
print(predict_data_file_path)
```
|Аргумент  |	Описание|
|----------|---------|
|avrg_sum	 | Среднее значение всех транзакций пользователя (средняя сумма sum за все время).|
|sum_sum	  |Общая сумма всех переводов пользователя (сумма всех транзакций в sum).|
|min_sum	  |Сумма всех отрицательных переводов пользователя (сумма всех отрицательных транзакций в sum).|
|max_sum	  |Сумма всех положительных переводов пользователя (сумма всех положительных транзакций в sum).|
|lrgst_sum	|Наибольшая сумма перевода пользователя (максимальное значение sum среди всех транзакций).|
|mnmm_sum	|Наименьшая сумма перевода пользователя (минимальное значение sum среди всех транзакций).|
|one_for_trnsctn |	Средняя сумма перевода на одну транзакцию (сумма sum деленная на количество переводов nmbr_of_trnsltns).|
|trnsctns_frnqnc	| Средний интервал между переводами в днях (раз в сколько дней были переводы, в среднем, по oprtn_date).|
|trnsctns_frnqnc_max |	Максимальный интервал (в днях) между транзакциями (самый большой перерыв по oprtn_date).|
|trnsctns_frnqnc_min	| Минимальный интервал (в днях) между транзакциями (самый маленький перерыв по oprtn_date).|
|frnqnt_trnsctn	| Наиболее частый тип движения денежных средств (1 или 0, по mvmnt_type).|
|frnqnt_trnsctn_1	| Количество транзакций с движением денежных средств 1 (количество единиц в mvmnt_type).|
|frnqnt_trnsctn_0	| Количество транзакций с движением денежных средств 0 (количество нулей в mvmnt_type).|

### Анализ работы кода

Этот код предназначен для предобработки и объединения данных клиентов и транзакций для построения модели прогнозирования досрочного выхода на пенсию. Основные этапы включают кодирование категориальных признаков, расчет показателей транзакционной активности и финансовой стабильности клиентов, агрегирование данных по клиентам, а также разделение на обучающую и тестовую выборки с балансировкой целевого класса. Скрипт завершает обработку сохранением подготовленных данных в CSV-файлы для последующего анализа. 
Для повышения эффективности при работе с большими массивами данных применяются методы, такие как поэтапное чтение транзакций чанками, оптимизация типов данных, и использование tqdm для отображения прогресса при агрегации данных.

## Построение и оценка модели предсказания раннего выхода на пенсию :pushpin:

```
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

```
### Анализ работы кода

Этот код использует библиотеку CatBoost для создания модели классификации, прогнозирующей вероятность раннего выхода на пенсию на основе характеристик клиентов. Данные загружаются из подготовленного файла, а затем делятся на обучающую и тестовую выборки. Модель CatBoostClassifier обучается с оптимизацией метрики точности и сохраняется для дальнейшего использования. После обучения точность модели проверяется на тестовых данных и дополнительных данных, предназначенных для предсказания. Итоговые предсказания сохраняются в CSV-файл для дальнейшего анализа.

## Результат работы кода 
```
0:  learn: 0.9992850  test: 0.9996425  best: 0.9996425 (0)  total: 63.3ms  remaining: 1m 3s
100:  learn: 1.0000000  test: 0.9999404  best: 0.9999404 (5)  total: 1.1s  remaining: 9.77s
200:  learn: 1.0000000  test: 0.9999404  best: 0.9999404 (5)  total: 2s  remaining: 7.97s
300:  learn: 1.0000000  test: 0.9999404  best: 0.9999404 (5)  total: 2.73s  remaining: 6.34s
400:  learn: 1.0000000  test: 0.9999404  best: 0.9999404 (5)  total: 3.51s  remaining: 5.25s
500:  learn: 1.0000000  test: 0.9999404  best: 0.9999404 (5)  total: 4.22s  remaining: 4.2s
600:  learn: 1.0000000  test: 0.9999404  best: 0.9999404 (5)  total: 5.03s  remaining: 3.34s
700:  learn: 1.0000000  test: 0.9999404  best: 0.9999404 (5)  total: 5.85s  remaining: 2.49s
800:  learn: 1.0000000  test: 0.9999404  best: 0.9999404 (5)  total: 6.7s  remaining: 1.66s
900:  learn: 1.0000000  test: 0.9999404  best: 0.9999404 (5)  total: 7.54s  remaining: 829ms
999:  learn: 1.0000000  test: 0.9999404  best: 0.9999404 (5)  total: 8.24s  remaining: 0us

bestTest = 0.9999404159
bestIteration = 5

Shrink model to first 6 iterations.
Точность модели на тестовых данных: 1.00
F1-Score: 1.00
Precision: 1.00
Recall: 1.00
Матрица путаницы:
 [[16294     0]
 [    1   488]]
```
```
Средняя точность при кросс-валидации: 1.00 ± 0.00
Точность модели на новых данных: 0.99
F1-Score на новых данных: 0.99
Precision на новых данных: 1.00
Recall на новых данных: 0.98
Матрица путаницы на новых данных:
 [[5986    0]
 [ 101 5885]]
```
