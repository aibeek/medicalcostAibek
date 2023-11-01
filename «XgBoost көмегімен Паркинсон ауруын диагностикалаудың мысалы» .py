import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Создаем синтетический датасет
data = {
    'Возраст': np.random.randint(40, 80, size=100),
    'Частота дрожания': np.random.normal(5, 2, size=100),
    'Тремор в покое': np.random.choice([0, 1], size=100),
    'Жесткость мышц': np.random.choice([0, 1], size=100),
    'Брадикинезия': np.random.choice([0, 1], size=100),
    'Инфекционные заболевания головного мозга': np.random.choice([0, 1], size=100),
    'Диагноз Паркинсона': np.random.choice([0, 1], size=100)
}

# Создаем DataFrame
df = pd.DataFrame(data)

# Увеличим вероятность болезни Паркинсона после инфекционных заболеваний головного мозга
df.loc[df['Инфекционные заболевания головного мозга'] == 1, 'Диагноз Паркинсона'] = np.random.choice([0, 1], 
                                                                                                   size=df[df['Инфекционные заболевания головного мозга'] == 1].shape[0], 
                                                                                                   p=[0.2, 0.8])

# Выделяем признаки (X) и целевую переменную (y)
X = df.drop(columns=['Диагноз Паркинсона'])
y = df['Диагноз Паркинсона']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем DMatrix для xgBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Задаем параметры модели
params = {
    'objective': 'binary:logistic',
    'colsample_bytree': 0.3,
    'learning_rate': 0.1,
    'max_depth': 5,
    'alpha': 10
}

# Обучаем модель
num_round = 100
bst = xgb.train(params, dtrain, num_round)

# Прогнозирование на тестовой выборке
y_pred = bst.predict(dtest)

# Преобразуем вероятности в бинарные предсказания
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

# Оценка качества модели
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy}')
