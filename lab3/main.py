
import keras
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
'''
# Sequential - модель сети, в которой группируется стек слоев
# Dence - слой сети. units - количество выходных каналов,
# input_shape - количество входных каналов
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1]),
])

x = tf.linspace(-1.0, 1.0, 100)
y = model.predict(x)

plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
w, b = model.weights
plt.title("Weight: {:0.2f}\n"
          "Bias: {:0.2f}".format(w[0][0], b[0]))
plt.show()
'''

'''
# варианты функции активации
# 'relu', 'elu', 'selu', 'swish'...
activation_layer = keras.layers.Activation('swish')

x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x)

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
'''
'''
# стек слоев с ReLU активацией
model = keras.Sequential([
    # скрытые слои
    keras.layers.Dense(units=4, activation='relu', input_shape=[2]),
    keras.layers.Dense(units=3, activation='relu'),
    # линейный выходной слой
    keras.layers.Dense(units=1),
])
# опримизатор и функция потерь
model.compile(
    optimizer="adam",
    loss="mae",
)
'''

'''
import pandas as pd
from IPython.display import display

sales = pd.read_csv('Sales.csv')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)

# Разбиваем на тренировочный и тестовый
test = sales.select_dtypes(exclude=["object", "bool"])
df_train = test.sample(frac=0.7, random_state=0)
df_valid = test.drop(df_train.index)
display(df_train.head(4))

# Приводим к промежутку [0, 1]
max_ = df_train.max(axis=0)
print(max_)
min_ = df_train.min(axis=0)
print(min_)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)
print(df_train.head())
print(df_valid.head())

# Проводим разбиение
X_train = df_train.drop('Total Profit', axis=1)
X_valid = df_valid.drop('Total Profit', axis=1)
y_train = df_train['Total Profit']
y_valid = df_valid['Total Profit']

print(X_train.shape)

from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[6]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
model.compile(
    optimizer='adam',
    loss='mae',
)
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=4,
)
print(history)

import pandas as pd

# конвертируем историю в DataFrame
history_df = pd.DataFrame(history.history)
# используем метод plot из pandas
history_df['loss'].plot()

import matplotlib.pyplot as plt
plt.show()
'''


# расширение сети
#model = keras.Sequential([
 #   keras.layers.Dense(16, activation='relu'),
#    keras.layers.Dense(1),
#])
#
#wider = keras.Sequential([
#    keras.layers.Dense(32, activation='relu'),
#    keras.layers.Dense(1),
#])
#
#deeper = keras.Sequential([
#    keras.layers.Dense(16, activation='relu'),
#    keras.layers.Dense(16, activation='relu'),
#    keras.layers.Dense(1),
#])



# ранняя остановка
import pandas as pd
from IPython.display import display

sales = pd.read_csv('Sales.csv')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)

# Разбиваем на тренировочный и тестовый
test = sales.select_dtypes(exclude=["object", "bool"])
df_train = test.sample(frac=0.7, random_state=0)
df_valid = test.drop(df_train.index)
display(df_train.head(4))

# приведение к промежутку [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

X_train = df_train.drop('Total Profit', axis=1)
X_valid = df_valid.drop('Total Profit', axis=1)
y_train = df_train['Total Profit']
y_valid = df_valid['Total Profit']

from tensorflow import keras
from keras import layers, callbacks

early_stopping = callbacks.EarlyStopping(
    min_delta=0.00001, # минимальное изменение, чтобы считаться за улучшение модели
    patience=20, # сколько эпох ждать до остановки
    restore_best_weights=True,
)

model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[6]),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])
wider = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[6]),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

deeper = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=[6]),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1),
])
def get_min_val(model):
    model.compile(
        optimizer='adam',
        loss='mae',
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=1024,
        epochs=500,
        callbacks=[early_stopping],
        verbose=0,
    )

    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot()
    print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

    import matplotlib.pyplot as plt
    plt.show()
get_min_val(model)
get_min_val(wider)
get_min_val(deeper)
