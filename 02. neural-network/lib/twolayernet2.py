# 신경망학습: 신경망에서의 기울기
import os
import sys
import numpy as np
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from layers import ReLU, Affine, SoftmaxWithLoss
except ImportError:
    print('Library Module Can Not Found')

params = dict()
layers = []


def initialize(sz_input, sz_hidden, sz_output, w_init=0.01):
    params['w1'] = w_init * np.random.randn(sz_input, sz_hidden)
    params['b1'] = np.zeros(sz_hidden)
    params['w2'] = w_init * np.random.randn(sz_hidden, sz_output)
    params['b2'] = np.zeros(sz_output)

    layers.append(Affine(params['w1'], params['b1']))
    layers.append(ReLU())
    layers.append(Affine(params['w2'], params['b2']))
    layers.append(SoftmaxWithLoss())


def forward_propagation(x, t=None):
    for layer in layers:
        x = layer.forward(x, t) if type(layer).__name__ == 'SoftmaxWithLoss' and t is not None else layer.forward(x)
    return x


def backward_propagation(dout):
    for layer in layers[::-1]:
        dout = layer.backward(dout)
    return dout


def loss(x, t):
    y = forward_propagation(x, t)
    return y


def accuracy(x, t):
    y = forward_propagation(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)

    acc = np.sum(y == t) / float(x.shape[0])
    return acc


def backpropagation_gradient_net(x, t):
    forward_propagation(x, t)
    backward_propagation(1)

    idxaffine = 0
    gradient = dict()

    for layer in layers:
        if type(layer).__name__ == 'Affine':
            idxaffine += 1
            gradient[f'w{idxaffine}'] = layer.dw
            gradient[f'b{idxaffine}'] = layer.db

    return gradient


def numerical_gradient_net(x, t):
    h = 1e-4
    gradient = dict()

    for key in params:
        param = params[key]
        param_gradient = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            temp = param[idx]

            param[idx] = temp + h
            h1 = loss(x, t)

            param[idx] = temp - h
            h2 = loss(x, t)

            param_gradient[idx] = (h1 - h2) / (2 * h)

            param[idx] = temp   # 값복원
            it.iternext()

        gradient[key] = param_gradient

    return gradient