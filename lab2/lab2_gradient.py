import pandas as pd
from sklearn.model_selection import train_test_split

path_to_data = "tornados.csv"
data = pd.read_csv(path_to_data)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)


#print(data.head())
# print(data.shape)
# data.info()
# print(data.describe())
# print(data.describe(include=["object", "bool"]))

size = data.shape
X = data.iloc[:int(size[0] * 0.8)].copy()
X_test_full = data.iloc[int(size[0] * 0.8):].copy()

#-------------------------------------------------------------------------------------------
# Удаляем прогнозируемый столбец из набора
X.dropna(axis=0, subset=["len"], inplace=True)
y = X["len"]
X.drop(["len"], axis=1, inplace=True)

# Разделяем на обучающий и тестовый набор
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Выбираем столбцы с низкой кардинальностью (категориальные)
low_cardinality_cols = [cname for cname in X_train_full.columns
                        if X_train_full[cname].nunique() < 30 and
                        X_train_full[cname].dtype == "object"]

# Выбираем числовые столбцы
numeric_cols = [cname for cname in X_train_full.columns
                if X_train_full[cname].dtype in ['int64', 'float64']]

# Оставляем только выбранные столбцы
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()
print(my_cols)
# Создаем фиктивные переменные для категориальных данных (аналог OneHotEncoder)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

from xgboost import XGBRegressor

# Создаем модель для градиентного спуска без параметров
my_model = XGBRegressor()
my_model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error

predictions = my_model.predict(X_valid)
print("MAE: " + str(mean_absolute_error(predictions, y_valid)))

my_model = XGBRegressor(n_estimators=500, max_depth=30)
my_model.fit(X_train, y_train)
predictions = my_model.predict(X_valid)
print("MAE: " + str(mean_absolute_error(predictions, y_valid)))
my_model = XGBRegressor(n_estimators=500, early_stopping_rounds=5)
my_model.fit(X_train, y_train,
             eval_set=[(X_valid, y_valid)])
predictions = my_model.predict(X_valid)
print("MAE: " + str(mean_absolute_error(predictions, y_valid)))
my_model = XGBRegressor(n_estimators=1000, n_jobs=4, early_stopping_rounds=5)
my_model.fit(X_train, y_train,
             eval_set=[(X_valid, y_valid)])
predictions = my_model.predict(X_valid)
print("MAE: " + str(mean_absolute_error(predictions, y_valid)))
