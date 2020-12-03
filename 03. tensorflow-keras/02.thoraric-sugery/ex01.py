# Thoraric Surgery Prediction Model(폐암 발병 예측 모델)
# Explore Dataset(데이터셋 탐색)

import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from matplotlib import pyplot as plt


dataset_file = './dataset/thoraric-surgery.csv'
df = pd.read_csv(dataset_file, header=None)
# print(df.info())
# print(df.head())

# 1.load training/test data
dataset = np.loadtxt(dataset_file, delimiter=',')
# print(dataset.shape)
x = np.array(dataset[:, 0:17])
t = np.array(dataset[:, 17])
print(x.shape, t.shape)

# 2. model frame config
model = Sequential()
model.add(Dense(30, input_dim= x.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. model fitting config
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. model fitting
history = model.fit(x, t, epochs=100, batch_size=10)

# 5. train loss
loss = history.history['loss']

# 6. result
result = model.evaluate(x, t)
print(f'\n(Loss, Accuracy) = {result[0], result[1]}')

# 7. predict (판단)