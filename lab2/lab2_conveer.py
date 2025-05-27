import pandas as pd
from sklearn.model_selection import train_test_split

path_to_data = "tornados.csv"
data = pd.read_csv(path_to_data)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
#
# for i in range(data.shape[0]//3):
#     data.at[i * 3, "Unit Cost"] = None
#     data.at[i * 3+1, "Unit Price"] = None
#     data.at[i * 3 + 2, "Units Sold"] = None



#print(data.head())
# print(data.shape)
# data.info()
# print(data.describe())
# print(data.describe(include=["object", "bool"]))



#-------------------------------------------------------------------------------------------
# Выбираем целевой столбец
y = data["len"]

X = data.drop(["len"], axis=1)

# Выбираем обучающий и тестовый набор
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Ищем категориальные столбцы
categorical_cols = [cname for cname in X_train_full.columns
                    if X_train_full[cname].nunique() < 30 and
                        X_train_full[cname].dtype == "object"]

# Выбираем столбцы с числовыми значениями
numerical_cols = [cname for cname in X_train_full.columns
                  if X_train_full[cname].dtype in ['int64', 'float64']]

# Оставляем только выбранные столбцы
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# print(X_train.head())

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Предобработка для числовых данных
numerical_transformer = SimpleImputer(strategy='most_frequent')

# Предобработка для категориальных данных
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Объединяем предобработку для числовых и категориальных данных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=5, random_state=0, max_depth=30)

from sklearn.metrics import mean_absolute_error

# Объединяем предобработку и моделирование
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Предобработка обучающих данных и обучение
my_pipeline.fit(X_train, y_train)

# Предобработка тестовых данных и прогнозирование
preds = my_pipeline.predict(X_valid)

# Оценка модели
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
