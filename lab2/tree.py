import pandas as pd
from sklearn.model_selection import train_test_split

data_file_path = 'tornados.csv'
wine_data = pd.read_csv(data_file_path)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 2000)

size = wine_data.shape
X = wine_data.iloc[:int(size[0]*0.8)].copy()
X_test_full = wine_data.iloc[int(size[0]*0.8):].copy()

# Удаляем прогнозируемый столбец из набора
X.dropna(axis=0, subset=['len'], inplace=True)
y = X["len"]
X.drop(["len"], axis=1, inplace=True)

# Разделяем на обучающий и тестовый набор
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Выбираем столбцы с низкой кардинальностью (категориальные)
low_cardinality_cols = [cname for cname in X_train_full.columns
                        if X_train_full[cname].nunique() <30 and
                        X_train_full[cname].dtype == "object"]
# Выбираем числовые столбцы
numeric_cols = [cname for cname in X_train_full.columns
                if X_train_full[cname].dtype in ['int64', 'float64']]

# Оставляем только выбранные столбцы
my_cols = low_cardinality_cols + numeric_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# Создаем фиктивные переменные для категориальных данных (аналог OneHotEncoder)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# decision tree
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
model=DecisionTreeRegressor(max_depth=9)
model=model.fit(X_train, y_train)
plt.figure(figsize=((20, 8)))
plot_tree(model,
          filled=True,
          feature_names=list(X_train.columns),
          class_names=list(X_train.columns),
          rounded=True)
plt.show()
scoreTree = model.predict(X_valid)
print("MAE: " + str(mean_absolute_error(scoreTree, y_valid)))
