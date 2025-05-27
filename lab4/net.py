import os, warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from keras import layers


def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


set_seed()

# РїР°СЂР°РјРµС‚СЂС‹ РґР»СЏ matplotlib
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore")

# Р·Р°РіСЂСѓР·РєР° РґР°РЅРЅС‹С…
ds_train_ = image_dataset_from_directory(
    "C:\\Users\\mailo\\Documents\\Р Р°Р±РѕС‚Р°\\РњРћ\\CV РїСЂРёРјРµСЂ\\train",
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    'valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)


def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

model = keras.Sequential([
    # РїРµСЂРІС‹Р№ СЃРІРµСЂС‚РѕС‡РЅС‹Р№ Р±Р»РѕРє
    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same',
                  # [РІС‹СЃРѕС‚Р°, С€РёСЂРёРЅР°, РєРѕР»РёС‡РµСЃС‚РІРѕ РєР°РЅР°Р»РѕРІ РґР»СЏ РєР°Р¶РґРѕРіРѕ РїРёРєСЃРµР»СЏ(RGB)]
                  input_shape=[128, 128, 3]),
    layers.MaxPool2D(),

    # РІС‚РѕСЂРѕР№ СЃРІРµСЂС‚РѕС‡РЅС‹Р№ Р±Р»РѕРє
    layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    # С‚СЂРµС‚РёР№ СЃРІРµСЂС‚РѕС‡РЅС‹Р№ Р±Р»РѕРє
    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    # СЃР°Рј РєР»Р°СЃСЃРёС„РёРєР°С‚РѕСЂ
    layers.Flatten(),
    layers.Dense(units=6, activation="relu"),
    layers.Dense(units=1, activation="sigmoid"),
])
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=40,
    verbose=0,
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
