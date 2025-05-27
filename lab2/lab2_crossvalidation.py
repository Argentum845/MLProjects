import pandas as pd

path_to_data = "tornados.csv"
data = pd.read_csv(path_to_data)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
#
# for i in range(data.shape[0]//3):
#     data.at[i * 3, "Unit Cost"] = None
#     data.at[i * 3+1, "Unit Price"] = None
#     data.at[i * 3 + 2, "Units Sold"] = None


#
# print(data.head())
# print(data.shape)
# print(data.info())



cols_to_use = ['fat', 'loss', 'wid', 'inj', 'ns']
X = data[cols_to_use]
y = data["len"]

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer(strategy="most_frequent")),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])

from sklearn.model_selection import cross_val_score

# Умножаем на -1, так как cross_val_score возвращает отрицательное MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE:\n", scores)

print("Среднее по всем экспериментам:")
print(scores.mean())
