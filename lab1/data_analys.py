import pandas as pd
import numpy as np

path_to_data = "tornados.csv"
tornados_data = pd.read_csv(path_to_data)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)
print(tornados_data.head())
print(tornados_data.shape)

print(tornados_data.columns)

print(tornados_data.info())

print(tornados_data.describe())
round_len = pd.Series(list(map(int, tornados_data['len'])))
tornados_data.insert(loc=len(tornados_data.columns), column='round_len', value=round_len)

print(pd.crosstab(tornados_data['round_len'], tornados_data['fat'], margins=True))
#print(tornados_data.groupby(['st'])[['st', 'fat']].describe())
