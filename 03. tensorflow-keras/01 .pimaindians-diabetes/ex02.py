# pimaindians-diabetes model fitting
# tensor-keras

import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from matplotlib import pyplot as plt

# 1.load training/test data
dataset = np.loadtxt('./dataset/pimaindians-diabetes.csv', delimiter=',')
train_x = dataset[:600, 0:8]
train_t = dataset[:600, 8]
test_x = dataset[600:, 0:8]
test_t = dataset[600:, 8]

# 2. model frame config
model = Sequential()
model.add(Dense(50, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. model fitting config
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
history = model.fit(train_x, train_t, validation_data=(test_x, test_t), epochs=200, batch_size=10)

# 5. train loss
train_loss = history.history['loss']
test_loss = history.history['val_loss']     # validation loss(test loss)

# 6. result
result = model.evaluate(train_x, train_t, verbose=0)  #verbose = 계산과정은 안보여준다.
print(f'\n(Loss, Accuracy) = ({result[0], result[1]})')

# 7. predict (판단)
data = np.array([[0, 118, 84, 47, 230, 45.8, 0.551, 31],
                 [6, 148, 72, 35, 0, 33.6, 0.627, 50],
                 [13, 106, 72, 54, 0, 36.6, 0.178, 45]
                 ])
predict = model.predict(data)
percentage = float(predict[2] * 100)

print(f'\n 당뇨 발병 확률 : {percentage:.2f}%')

# # 8. graph
# xlen = np.arange(len(train_loss))
#
# plt.plot(xlen, train_loss, marker='.', c='blue', label='Train Loss')
# plt.plot(xlen, test_loss, marker='.', c='red', label='Test Loss')
#
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='best')
#
# plt.show()