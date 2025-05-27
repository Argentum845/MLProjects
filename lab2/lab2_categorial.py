import pandas as pd
from sklearn.model_selection import train_test_split

path_to_data = "Sales.csv"
data = pd.read_csv(path_to_data)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)

for i in range(data.shape[0]//3):
    data.at[i * 3, "Unit Cost"] = None
    data.at[i * 3+1, "Unit Price"] = None
    data.at[i * 3 + 2, "Units Sold"] = None



#print(data.head())
print(data.shape)
print(data.info())
print(data.describe())
print(data.describe(include=["object", "bool"]))



#-------------------------------------------------------------------------------------------
# Выбираем целевой столбец
y = data["Total Cost"]

X = data.drop(["Total Cost"], axis=1)

# Разделяем на обучающий и тестовый набор
X_train_full, X_valid_full, y_train, y_valid =\
    train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Удаляем столбцы с недостающими значениями (самый простой подход)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()] 
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# "Кардинальность" - число уникальных значений в столбце
# Выбираем столбцы с относительно низкой кардинальностью
low_cardinality_cols = [cname for cname in X_train_full.columns
                        if X_train_full[cname].nunique() < 15 and
                        X_train_full[cname].dtype == "object"]
print(low_cardinality_cols)
# Выбираем столбцы с числовыми значениями
numerical_cols = [cname for cname in X_train_full.columns
                  if X_train_full[cname].dtype in ['int64', 'float64']]

# Оставляем только выбранные столбцы
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

print(X_train.head())

# Получаем список столбцов с категориальными данными
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Столбцы с категориальными данными:")
print(object_cols)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Функция для сравнения эффективности разных подходов
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE при первом подходе (Удаление категориальных данных):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

from sklearn.preprocessing import OrdinalEncoder

# Создаем копию, чтобы не испортить исходные данные
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Применяем упорядоченную кодировку к категориальным данным
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print("MAE при втором подходе (Упорядоченное кодирование):")
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

from sklearn.preprocessing import OneHotEncoder

# Применяем прямое кодирование для каждого столбца с категориальными данными
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# Возвращаем индексы
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Удаляем категориальные столбцы (заменим их на закодированные)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Добавляем закодированные столбцы
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Во избежание ошибок приводим все имена столбцов к строковому типу
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("MAE при третьем подходе (Прямое кодирование):")
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
