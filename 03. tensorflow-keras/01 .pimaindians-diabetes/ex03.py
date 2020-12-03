# pimaindians-diabetes model fitting
# TwolayerNet2

# 1.load training/test data
import numpy as np

dataset = np.loadtxt('./dataset/pimaindians-diabetes.csv', delimiter=',')
train_x = dataset[:, 0:8]
train_t = dataset[:, 8]