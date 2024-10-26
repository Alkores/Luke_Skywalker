# Luke_Skywalker
Проект представлен командой Luke_Skywalker для окружного хакатона ПФО.
```
    /' '\       / " \
   |  ,--+-----4 /   |
   ',/   o  o     --.;
--._|_   ,--.  _.,-- \----.
------'--`--' '-----,' VJ  |
     \_  ._\_.   _,-'---._.'
       `--...--``  /
         /###\   | |
         |.   `.-'-'.
        .||  /,     |
       do_o00oo_,.ob
```
## Кейс
Прогнозирование раннего выхода на пенсию.

### Кейсодержатель
АО «Негосударственный пенсионный фонд «БУДУЩЕЕ»
## Описание проекта
### Проблема
Современные государственные пенсионные системы сталкиваются с вызовами, обусловленными изменением демографической структуры. Увеличение продолжительности жизни и снижение рождаемости увеличивают финансовую нагрузку, так как государственные пенсионные системы должны выплачивать пенсии дольше. С другой стороны, для негосударственных пенсионных фондов (НПФ) ранний выход на пенсию клиентов уменьшает объем взносов и чистую прибыль, поскольку сокращает время для инвестиционной активности.

### Цель проекта
Создание модели машинного обучения, которая оценит вероятность досрочного выхода на пенсию клиента ОПС или НПФ. Модель позволит более точно прогнозировать денежные потоки и финансовые обязательства для оптимального планирования.

 ## Задача
Задача проекта — построить модель, способную предсказывать, обратится ли клиент за досрочной пенсией, основываясь на его характеристиках и макроэкономических показателях. Это знание позволяет государственным и негосударственным пенсионным фондам:

- учитывать риски и планировать финансовые обязательства,
- адаптировать инвестиционные стратегии,
- укреплять устойчивость пенсионной системы.

## Код 1

```
import pandas as pd
from sklearn.model_selection import train_test_split

# Укажите пути к исходным файлам
clients_file_path = r"D:\Download\train_dataset_train_data_NPF\train_data\cntrbtrs_clnts_ops_trn.csv"
transactions_file_path = r"D:\Download\train_dataset_train_data_NPF\train_data\trnsctns_ops_trn.csv"
# Укажите путь для сохранения новой таблицы
train_data_file_path = r"D:\Download\train_dataset_train_data_NPF\train_data\train_data.csv"
predict_data_file_path = r"D:\Download\train_dataset_train_data_NPF\train_data\predict_data.csv"

# Чтение файлов CSV
clients_df = pd.read_csv(clients_file_path, sep=';', encoding='ANSI')
transactions_df = pd.read_csv(transactions_file_path, sep=';', encoding='utf-8')

# Кодирование пола: 'м' = 1, 'ж' = 0
clients_df['gndr'] = clients_df['gndr'].replace({'м': 1, 'ж': 0})

# Кодирование статуса счета: 'Накопительный период' = 1, 'Выплатной период' = 0
clients_df['accnt_status'] = clients_df['accnt_status'].replace({'Накопительный период': 1, 'Выплатной период': 0})

# Удаление строк с null значениями в таблице клиентов
clients_df.dropna(inplace=True)

# Объединение таблиц по 'accnt_id'
merged_df = pd.merge(clients_df, transactions_df, on='accnt_id', how='left')

# Удаляем строки с null значениями в транзакциях
merged_df.dropna(subset=['sum'], inplace=True)

# Вычисление количества переводов для каждого клиента
nmbr_of_trnsltns = merged_df.groupby('clnt_id').size().reset_index(name='nmbr_of_trnsltns')

# Приводим тип к целочисленному после вычисления
nmbr_of_trnsltns['nmbr_of_trnsltns'] = nmbr_of_trnsltns['nmbr_of_trnsltns'].astype(int)

# Создаем новую колонку для оценки
merged_df['transaction_value'] = merged_df.apply(
    lambda x: x['sum'] if x['mvmnt_type'] == 1 else -x['sum'], axis=1
)

# Удаляем нулевые переводы
filtered_transactions = merged_df[merged_df['sum'] != 0]

# Вычисляем общую положительность переводов
# Используем только положительные и отрицательные суммы
positive_sum = filtered_transactions[filtered_transactions['transaction_value'] > 0].groupby('clnt_id')['transaction_value'].sum().reset_index(name='positive_sum')
negative_sum = filtered_transactions[filtered_transactions['transaction_value'] < 0].groupby('clnt_id')['transaction_value'].sum().reset_index(name='negative_sum')

# Объединяем положительные и отрицательные суммы
trnsltn_vltn = pd.merge(positive_sum, negative_sum, on='clnt_id', how='left').fillna(0)

# Рассчитываем trnsltn_vltn как нормализованное значение от 0 до 1
trnsltn_vltn['trnsltn_vltn'] = trnsltn_vltn.apply(
    lambda x: round((x['positive_sum'] / (x['positive_sum'] - x['negative_sum'] + 1e-10)), 2) if (x['positive_sum'] + abs(x['negative_sum'])) > 0 else 0, axis=1
)

# Удаляем клиентов без переводов
trnsltn_vltn = trnsltn_vltn[(trnsltn_vltn['positive_sum'] > 0) | (trnsltn_vltn['negative_sum'] > 0)]

# Объединяем данные по количеству переводов и положительности переводов
result_df = pd.merge(nmbr_of_trnsltns, trnsltn_vltn[['clnt_id', 'trnsltn_vltn']], on='clnt_id', how='left')

# Убедимся, что nmbr_of_trnsltns является целым числом
result_df['nmbr_of_trnsltns'] = result_df['nmbr_of_trnsltns'].astype(int)

# Объединяем с клиентами
final_df = pd.merge(clients_df[['clnt_id', 'gndr', 'brth_yr', 'prsnt_age', 'erly_pnsn_flg', 'accnt_status', 'pnsn_age']],
                    result_df, on='clnt_id', how='left')

# Удаляем строки, где trnsltn_vltn является NaN
final_df.dropna(subset=['trnsltn_vltn'], inplace=True)

# Разделение данных на обучающую и тестовую выборки (70% на 30%)
train_df, predict_df = train_test_split(final_df, test_size=0.3, random_state=42)

# Удаляем столбец erly_pnsn_flg у предикта
predict_df.drop(columns=['erly_pnsn_flg'], inplace=True)

# Сохранение новых таблиц в CSV
train_df.to_csv(train_data_file_path, index=False, encoding='utf-8', sep=';')
predict_df.to_csv(predict_data_file_path, index=False, encoding='utf-8', sep=';')

print("Новые таблицы успешно созданы и сохранены по пути:")
print(train_data_file_path)
print(predict_data_file_path)
```
### Вывод

## Модель обучения

```
# Импорт библиотек
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка данных из CSV
data = pd.read_csv("train_data.csv", encoding='utf-8', sep=';')  # Входной файл с данными для обучения

# Подготовка данных
feature_columns = ['gndr', 'brth_yr', 'prsnt_age', 'accnt_status', 'pnsn_age', 'nmbr_of_trnsltns', 'trnsltn_vltn']
X = data[feature_columns]  # Признаки
y = data['erly_pnsn_flg']  # Целевая переменная

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, eval_metric='Accuracy', verbose=100)
model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

# Сохранение модели
model.save_model("catboost_model.cbm")  # Сохранение модели в файл

# Оценка модели на тестовых данных
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели на тестовых данных: {accuracy:.2f}")

# Прогноз для новых данных (входной файл с эталоном)
new_data = pd.read_csv("predict1_data.csv", encoding='utf-8', sep=';')  # Входной файл с данными для предсказания

# Предполагаем, что в new_data есть столбец 'erly_pnsn_flg' для сверки
if 'erly_pnsn_flg' in new_data.columns:
    y_true = new_data['erly_pnsn_flg']  # Истинные значения
    new_data_predictions = model.predict(new_data[feature_columns])  # Прогноз

    # Оценка модели на новых данных
    new_accuracy = accuracy_score(y_true, new_data_predictions)
    print(f"Точность модели на новых данных: {new_accuracy:.2f}")

    # Сохранение предсказаний в CSV
    output = new_data.copy()
    output['predicted_erly_pnsn_flg'] = new_data_predictions  # Добавление предсказаний в выходной файл
    output.to_csv("VALID_data.csv", index=False)  # Выходной файл с предсказаниями
else:
    print("В файле предсказаний отсутствует столбец 'erly_pnsn_flg' для сверки.")
```

// Написать вывод на коды
