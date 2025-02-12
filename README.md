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


  <img src= "https://i.imgur.com/Y4JublQ.jpg" title="Jupyter Notebook" width="120" height="100"/>&nbsp; <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original-wordmark.svg" title="Python" width="100" height="100"/>&nbsp; 
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/keras/keras-original.svg" title="Keras" width="100" height="100"/>&nbsp;
           <img src="https://i.imgur.com/DZ0ilNv.png?1" title="CatBoost" width="120" height="100"/>&nbsp;


## Подготовка данных обработка клиентов и транзакций :scissors:

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

### Анализ работы кода Стандартизации

Этот код предназначен для предобработки и объединения данных клиентов и транзакций для построения модели прогнозирования досрочного выхода на пенсию. Основные этапы включают кодирование категориальных признаков, расчет показателей транзакционной активности и финансовой стабильности клиентов, агрегирование данных по клиентам, а также разделение на обучающую и тестовую выборки с балансировкой целевого класса. Скрипт завершает обработку сохранением подготовленных данных в CSV-файлы для последующего анализа. 
Для повышения эффективности при работе с большими массивами данных применяются методы, такие как поэтапное чтение транзакций чанками, оптимизация типов данных, и использование tqdm для отображения прогресса при агрегации данных.

## Построение и оценка модели предсказания раннего выхода на пенсию :pushpin:

### Анализ работы кода CatBoost

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
## Keras 

В ходе работы было принята идея перейти на Keras из-за его удобства, модульности, расширяемости и простоты использования. Keras предоставляет единый и понятный API, упрощая процесс создания и обучения нейронных сетей. Благодаря модульной архитектуре, каждый компонент модели (слои, функции активации, оптимизаторы) настраивается и комбинируется, что позволяет гибко адаптировать модель под задачу. Также, Keras легко расширяется, что позволяет добавлять собственные модули и функции для кастомизации. Код на Python делает модели читаемыми и компактными, ускоряя разработку и улучшая понимание работы моделей даже для новичков.

```
Матрица путаницы:
[[5986    0]
 [  48 5938]]

Отчет о классификации:
              precision    recall  f1-score   support

           0       0.99      1.00      1.00      5986
           1       1.00      0.99      1.00      5986

    accuracy                           1.00     11972
   macro avg       1.00      1.00      1.00     11972
weighted avg       1.00      1.00      1.00     11972

F1-Score: 1.00
Precision: 1.00
Recall: 0.99
```

## Результат работы тестового кода
```
|Обработка групп: 100%|██████████| 94029/94029 [02:04<00:00, 754.00it/s]
Новая таблица успешно создана и сохранена по пути:
D:\Download\test_dataset_test_data_NPF\test_data\train1_data.csv
Epoch 1/100
D:\Anaconda\envs\NewEnv\lib\site-packages\keras\src\layers\core\dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 3s 984us/step - accuracy: 0.9900 - loss: 0.0574 - val_accuracy: 0.9996 - val_loss: 0.0012
Epoch 2/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 944us/step - accuracy: 0.9988 - loss: 0.0046 - val_accuracy: 0.9995 - val_loss: 0.0014
Epoch 3/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 969us/step - accuracy: 0.9996 - loss: 0.0014 - val_accuracy: 0.9998 - val_loss: 7.7695e-04
Epoch 4/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 927us/step - accuracy: 0.9994 - loss: 0.0023 - val_accuracy: 0.9998 - val_loss: 8.2601e-04
Epoch 5/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9995 - loss: 0.0014 - val_accuracy: 0.9995 - val_loss: 0.0011
Epoch 6/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 983us/step - accuracy: 0.9996 - loss: 0.0016 - val_accuracy: 0.9998 - val_loss: 5.3372e-04
Epoch 7/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 982us/step - accuracy: 0.9995 - loss: 0.0010 - val_accuracy: 0.9996 - val_loss: 0.0012
Epoch 8/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 953us/step - accuracy: 0.9997 - loss: 8.9315e-04 - val_accuracy: 0.9996 - val_loss: 8.0746e-04
Epoch 9/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 920us/step - accuracy: 0.9997 - loss: 9.3055e-04 - val_accuracy: 0.9998 - val_loss: 0.0021
Epoch 10/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 944us/step - accuracy: 0.9997 - loss: 0.0011 - val_accuracy: 0.9998 - val_loss: 0.0014
Epoch 11/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 912us/step - accuracy: 0.9997 - loss: 0.0011 - val_accuracy: 0.9998 - val_loss: 0.0018
Epoch 12/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 920us/step - accuracy: 0.9996 - loss: 0.0015 - val_accuracy: 0.9998 - val_loss: 9.6404e-04
Epoch 13/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9996 - loss: 0.0013 - val_accuracy: 0.9998 - val_loss: 0.0014
Epoch 14/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9997 - loss: 6.1174e-04 - val_accuracy: 0.9996 - val_loss: 8.9930e-04
Epoch 15/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 934us/step - accuracy: 0.9997 - loss: 8.6523e-04 - val_accuracy: 0.9998 - val_loss: 0.0017
Epoch 16/100
2098/2098 ━━━━━━━━━━━━━━━━━━━━ 2s 949us/step - accuracy: 0.9997 - loss: 0.0012 - val_accuracy: 0.9998 - val_loss: 0.0012
2611/2611 ━━━━━━━━━━━━━━━━━━━━ 1s 523us/step

Предсказания для нового набора данных:
       erly_pnsn_flg
0                  0
1                  0
2                  0
3                  0
4                  0
...              ...
83519              0
83520              0
83521              0
83522              0
83523              0

[83524 rows x 1 columns]

```
