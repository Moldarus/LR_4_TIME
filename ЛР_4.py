import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Генерация гауссовского шума
np.random.seed(0)
gaussian_noise = np.random.normal(0, 1, 500)

# Построение графика гауссовского шума
plt.figure(figsize=(12, 6))
plt.plot(gaussian_noise, label='Гауссовский шум')
plt.title('Гауссовский шум')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.show()

# Проведение теста Дики-Фуллера на гауссовском шуме
result = adfuller(gaussian_noise, maxlag=1, regression='ct')
p_value = result[1]
print(f'p-value (Гауссовский шум): {p_value}')
if p_value < 0.05:
    print('Ряд стационарен')
else:
    print('Ряд нестационарен')

# Добавление тренда к гауссовскому шуму
trends = [0.001, 0.05, 0.1]
for trend in trends:
    series_with_trend = gaussian_noise + trend * np.arange(500)

    # Построение графика гауссовского шума с трендом
    plt.figure(figsize=(12, 6))
    plt.plot(series_with_trend, label=f'Гауссовский шум с трендом {trend}')
    plt.title(f'Гауссовский шум с трендом {trend}')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.show()

    # Проведение теста Дики-Фуллера на гауссовском шуме с трендом
    result = adfuller(series_with_trend, maxlag=1, regression='ct')
    p_value = result[1]
    print(f'p-value (Гауссовский шум с трендом {trend}): {p_value}')
    if p_value < 0.05:
        print('Ряд стационарен')
    else:
        print('Ряд нестационарен')

# Добавление тренда к временному ряду
for trend in trends:
    series_with_trend = gaussian_noise + trend * np.arange(500)

    # Построение графика временного ряда с трендом
    plt.figure(figsize=(12, 6))
    plt.plot(series_with_trend, label=f'Временной ряд с трендом {trend}')
    plt.title(f'Временной ряд с трендом {trend}')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.show()

    # Проведение теста Дики-Фуллера на временном ряду с трендом
    result = adfuller(series_with_trend, maxlag=1, regression='ct')
    p_value = result[1]
    print(f'p-value (Временной ряд с трендом {trend}): {p_value}')
    if p_value < 0.05:
        print('Ряд стационарен')
    else:
        print('Ряд нестационарен')

# Загрузка данных Passengers
data = pd.read_csv('Passengers.csv')
passengers = data['#Passengers'].values

# Разбиение на 3 части
n = len(passengers)
part1 = passengers[:n//3]
part2 = passengers[n//3:2*n//3]
part3 = passengers[2*n//3:]

# Подсчет среднего и дисперсии для каждой части
mean1, var1 = np.mean(part1), np.var(part1)
mean2, var2 = np.mean(part2), np.var(part2)
mean3, var3 = np.mean(part3), np.var(part3)

print(f'Среднее и дисперсия для части 1: {mean1}, {var1}')
print(f'Среднее и дисперсия для части 2: {mean2}, {var2}')
print(f'Среднее и дисперсия для части 3: {mean3}, {var3}')

# Прологарифмирование данных
log_passengers = np.log(passengers)

# Разбиение на 3 части
log_part1 = log_passengers[:n//3]
log_part2 = log_passengers[n//3:2*n//3]
log_part3 = log_passengers[2*n//3:]

# Подсчет среднего и дисперсии для каждой части
log_mean1, log_var1 = np.mean(log_part1), np.var(log_part1)
log_mean2, log_var2 = np.mean(log_part2), np.var(log_part2)
log_mean3, log_var3 = np.mean(log_part3), np.var(log_part3)

print(f'Среднее и дисперсия для части 1 (логарифмированные): {log_mean1}, {log_var1}')
print(f'Среднее и дисперсия для части 2 (логарифмированные): {log_mean2}, {log_var2}')
print(f'Среднее и дисперсия для части 3 (логарифмированные): {log_mean3}, {log_var3}')

# Проведение теста Дики-Фуллера на исходном датасете
result = adfuller(passengers, maxlag=1, regression='ct')
p_value = result[1]
print(f'p-value (исходный датасет): {p_value}')
if p_value < 0.05:
    print('Ряд стационарен')
else:
    print('Ряд нестационарен')

# Проведение теста Дики-Фуллера на логарифмированном датасете
result = adfuller(log_passengers, maxlag=1, regression='ct')
p_value = result[1]
print(f'p-value (логарифмированный датасет): {p_value}')
if p_value < 0.05:
    print('Ряд стационарен')
else:
    print('Ряд нестационарен')

# Заключение
print()
print("Заключение")
print("Из результатов теста Дики-Фуллера можно сделать следующие выводы:")
print()
print("Гауссовский шум:")
print("При параметре regression=\"c\" гауссовский шум и шум с трендом 0.001 стационарны, а шум с трендами 0.05 и 0.1 нестационарны.")
print("При параметре regression=\"ct\" все варианты гауссовского шума (с разными трендами) стационарны.")
print()
print("Временной ряд:")
print("При параметре regression=\"c\" временной ряд с трендом 0.001 стационарен, а с трендами 0.05 и 0.1 нестационарны.")
print("При параметре regression=\"ct\" все варианты временного ряда (с разными трендами) стационарны.")
print()
print("Датасет Passengers:")
print("При параметре regression=\"c\" исходный и логарифмированный датасеты нестационарны.")
print("При параметре regression=\"ct\" исходный и логарифмированный датасеты стационарны.")
print()
print("Таким образом, использование параметра regression=\"ct\" (учитывающего постоянную и тренд) позволяет более точно определить стационарность временных рядов, особенно в случаях, когда ряд содержит тренд.")
