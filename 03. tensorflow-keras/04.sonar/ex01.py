# Sonar Mineral Binary Classification Model(초음파 광물 예측 모델)
# Explore Dataset(데이터 탐색)
import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from matplotlib import pyplot as plt

# 1. Load training/test data
dataset_file = './dataset/sonar.csv'
df = pd.read_csv(dataset_file, header=None)
print(df.info())
print(df.head())