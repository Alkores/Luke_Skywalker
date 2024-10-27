import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# Укажите пути к исходным файлам
clients_file_path = r"D:\Download\test_dataset_test_data_NPF\test_data\cntrbtrs_clnts_ops_tst.csv"
transactions_file_path = r"D:\Download\test_dataset_test_data_NPF\test_data\trnsctns_ops_tst.csv"
# Укажите путь для сохранения новой таблицы
train_data_file_path = r"D:\Download\test_dataset_test_data_NPF\test_data\train1_data.csv"

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
for chunk in pd.read_csv(transactions_file_path, sep=';', encoding='Windows-1251', chunksize=100000, dtype={
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
final_df = pd.merge(clients_df[['clnt_id', 'gndr', 'brth_yr', 'prsnt_age', 'accnt_status', 'pnsn_age']],
                    aggregated_data, on='clnt_id', how='left')

# Удаляем строки, где 'trnsltn_vltn' является NaN
final_df.dropna(subset=['trnsltn_vltn'], inplace=True)

# Удаляем строки с нулевыми значениями в 'trnsltn_vltn' после формирования таблицы
final_df = final_df[final_df['trnsltn_vltn'] != 0]

# Сохранение новых таблиц в CSV
final_df.to_csv(train_data_file_path, index=False, encoding='utf-8', sep=';')

print("Новая таблица успешно создана и сохранена по пути:")
print(train_data_file_path)

# Загрузка данных для обучения
train_data_path = r"D:\Download\train_dataset_train_data_NPF\train_data\train1_data.csv"  # Путь к вашему обучающему набору данных
new_data_path = train_data_file_path  # Используем файл с новыми данными для предсказания

# Загрузка данных для обучения
train_data = pd.read_csv(train_data_path, encoding='utf-8', sep=';')

# Проверка на пропущенные значения
if train_data.isnull().values.any():
    print("В данных есть пропущенные значения. Пожалуйста, обработайте их перед обучением.")
    exit()

# Подготовка данных для обучения
feature_columns = ['gndr', 'brth_yr', 'prsnt_age', 'accnt_status', 'pnsn_age', 
                   'avrg_sum', 'sum_sum', 'min_sum', 'max_sum', 'lrgst_sum', 
                   'mnmm_sum', 'one_for_trnsctn', 'trnsctns_frnqnc', 
                   'trnsctns_frnqnc_max', 'trnsctns_frnqnc_min', 
                   'frnqnt_trnsctn', 'frnqnt_trnsctn_1', 'frnqnt_trnsctn_0', 
                   'nmbr_of_trnsltns', 'trnsltn_vltn']
X_train = train_data[feature_columns]
y_train = train_data['erly_pnsn_flg']  # Предполагается, что этот столбец присутствует в обучающем наборе данных

# Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Создание модели
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dropout(0.5))  # Слой Dropout для предотвращения переобучения
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Для бинарной классификации

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Определение колбэков
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# Обучение модели
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Подготовка данных для предсказания
new_data = pd.read_csv(r"D:\Download\test_dataset_test_data_NPF\test_data\train1_data.csv", encoding='utf-8', sep=';')  # Загрузка нового набора данных для предсказания

# Проверка на пропущенные значения в новом наборе данных
if new_data.isnull().values.any():
    print("В новом наборе данных есть пропущенные значения. Пожалуйста, обработайте их перед предсказанием.")
    exit()

# Подготовка данных для предсказания
X_new = new_data[feature_columns]

# Стандартизация новых данных
X_new_scaled = scaler.transform(X_new)

# Предсказание
y_new_pred = (model.predict(X_new_scaled) > 0.5).astype("int32")

# Создание нового столбца erly_pnsn_flg и добавление предсказаний
new_data['erly_pnsn_flg'] = y_new_pred  # Добавление предсказаний в новый DataFrame

# Сохранение нового DataFrame с предсказаниями в CSV файл
output_path = r"D:\Download\test_dataset_test_data_NPF\test_data\predictions.csv"  # Путь для сохранения файла с предсказаниями
new_data.to_csv(output_path, index=False, sep=';')  # Сохранение в CSV файл

print("\nПредсказания для нового набора данных:")
print(new_data[['erly_pnsn_flg']])
