
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings


def task(kernel):
    plt.rc('figure', autolayout=True)
    plt.rc('axes', labelweight='bold', labelsize='large',
           titleweight='bold', titlesize=18, titlepad=10)
    plt.rc('image', cmap='magma')
    warnings.filterwarnings("ignore")

    image_path = 'mushroom.jpg'
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)

    plt.figure(figsize=(6, 6))
    plt.imshow(tf.squeeze(image), cmap='gray')
    plt.axis('off')
    plt.show()

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)
    print(image.shape)
    image = tf.reshape(image, [*image.shape, 1, 1])
    kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
    kernel = tf.cast(kernel, dtype=tf.float32)

    image_filter = tf.nn.conv2d(
        input=image,
        filters=kernel,
        # чуть позже о последних двух параметрах
        strides=1,
        padding='SAME',
    )
    image_filter = tf.reshape(image_filter, [*image.shape[:4]])
    print(image_filter.shape)
    # активация
    image_detect = tf.nn.relu(image_filter)
    print(image_detect.shape)
    # отображение промежуточного результата
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.imshow(tf.squeeze(image), cmap='gray')
    plt.axis('off')
    plt.title('Input')
    plt.subplot(132)
    plt.imshow(tf.squeeze(image_filter))
    plt.axis('off')
    plt.title('Filter')
    plt.subplot(133)
    plt.imshow(tf.squeeze(image_detect))
    plt.axis('off')
    plt.title('Detect')
    plt.show()

    # применение пуллинга
    image_condense = tf.nn.pool(
        input=image_detect, # предобработанное изображение
        window_shape=(2, 2),
        pooling_type='MAX',
        strides=(2, 2),
        padding='SAME',
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(tf.squeeze(image_condense))
    plt.axis('off')
    plt.show()


'''
kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
], dtype=tf.float32)
task(kernel)

kernel = tf.constant([
    [1, 1, 1],
    [1,  8, 1],
    [1, 1, 1],
], dtype=tf.float32)
task(kernel)


kernel = tf.constant([
    [0, 1, 2],
    [2,  2, 0],
    [0, 1, 2],
], dtype=tf.float32)
task(kernel)

kernel = tf.constant([
    [1, 0, 1],
    [0,  2, 0],
    [3, 0, 2],
], dtype=tf.float32)
task(kernel)
'''
kernel = tf.constant([
    [-1, 1, -1],
    [1,  1, 1],
    [-1, 1, -1],
], dtype=tf.float32)
task(kernel)


kernel = tf.constant([
    [0, 1, 0],
    [0,  1, 0],
    [0, 1, 0],
], dtype=tf.float32)
task(kernel)
