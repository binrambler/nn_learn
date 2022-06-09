"""
Построение и обучение точно такой же нейросети, но с помощью tensorflow
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import backend
import matplotlib.pyplot as plt

# 0.6, 0.7 - входные значения сети, 0.9 - выходное
ds = np.array([[0.6, 0.7, 0.9]])
x_train = ds[0, :2]
y_train = ds[0, 2]
# подготавливам размерность данных:
# x_train (2, ) > (1, 2),
# y_train () > (1, )
x_train = np.expand_dims(x_train, axis=0)
y_train = np.expand_dims(y_train, axis=0)
# строим сеть
model = models.Sequential()
# 1 внутренний слой
model.add(layers.Dense(3, activation='sigmoid', input_shape=(2, )))
# 2 внутренний слой
model.add(layers.Dense(2, activation='sigmoid'))
# 3 выходной слой
model.add(layers.Dense(1, activation='sigmoid'))
# model.summary()
# оптимизатор со скоростью обучения 0.01
adam = tf.optimizers.Adam(learning_rate=0.1)
model.compile(loss='mean_squared_error',
              optimizer=adam,
              metrics=['mae'])
# можно загрузить веса, рассчитанные в предыдущем обучении
# model.load_weights('d:/py/nn.h5')
history = model.fit(x_train, y_train, epochs=200)

# для интереса можно посмотреть значение на выходе каждого слоя
# смотрим на выходе
get_layer_output = backend.function(inputs = model.layers[0].input, outputs = model.layers[2].output)
print(get_layer_output([x_train])[0])

# можно сохранить веса
# model.save_weights('d:/py/nn.h5')

# строим функцию средней абсолютной ошибки
plt.plot(history.history['mae'])
plt.xlabel('Epoch')
plt.ylabel('Mean absolute error')
plt.show()

