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

# URL для CSV файлов
train_data_url = r"D:\Download\train_dataset_train_data_NPF\train_data\train1_data.csv"  # Замените на URL вашего обучающего набора данных
test_data_url = r"D:\Download\train_dataset_train_data_NPF\train_data\predict1_data.csv"      # Замените на URL вашего тестового набора данных

# Загрузка данных для обучения
train_data = pd.read_csv(train_data_url, encoding='utf-8', sep=';')

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
y_train = train_data['erly_pnsn_flg']

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

# Загрузка данных для тестирования
test_data = pd.read_csv(test_data_url, encoding='utf-8', sep=';')  # Замените на URL вашего тестового набора данных

# Проверка на пропущенные значения в тестовых данных
if test_data.isnull().values.any():
    print("В тестовых данных есть пропущенные значения. Пожалуйста, обработайте их перед оценкой.")
    exit()

# Подготовка данных для тестирования
X_test = test_data[feature_columns]
y_test = test_data['erly_pnsn_flg']

# Стандартизация тестовых данных
X_test_scaled = scaler.transform(X_test)

# Оценка модели
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Вывод метрик
print("Матрица путаницы:")
print(confusion_matrix(y_test, y_pred))

print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred))

# Дополнительные метрики
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"F1-Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Визуализация обучения
plt.plot(history.history['accuracy'], label='Тренировочная точность')
plt.plot(history.history['val_accuracy'], label='Валидационная точность')
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпоха')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Тренировочная потеря')
plt.plot(history.history['val_loss'], label='Валидационная потеря')
plt.title('Потеря модели')
plt.ylabel('Потеря')
plt.xlabel('Эпоха')
plt.legend()
plt.show()
