"""
perceptron and multilayer perceptron implementation
Lucas Pires Camargo, 2022
Tópicos Especiais em Sistemas Eletrônicos IV - Aprendizado de Máquina
Programa de Pós-Graduação em Engenharia de Sistemas Eletrônicos – PPGESE


This script exists only for generating a simple linearly-separable dataset.
Its output is what goes in the 'data' file.
It can be used with the 'iris' dataset source.
"""

from cmath import cos, pi, sin
from random import random
import numpy
from matplotlib import pyplot as plt

NUM_POINTS = 1000
SCALE = 1.0
x = []
y = []
klass = []
all = []
all_1 = []
all_2 = []

# LINE PARAMETERS
la = -1.0
lb = 0.35

for i in range(NUM_POINTS):
    magnitude = random()
    angle = 2*pi*random()
    px = magnitude * cos(angle).real
    py = magnitude * sin(angle).real

    x.append(px)
    y.append(py)
    klass.append(1 if ((px*la+lb) <= py) else 2)
    all.append((x[-1],y[-1],klass[-1],))
    (all_1 if klass[-1] == 1 else all_2).append(all[-1])


plt.scatter(x, y, c=klass)
plt.show()

for tuple in all:
    print(f'{tuple[0]},{tuple[1]},{tuple[2]}')