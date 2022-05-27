"""
perceptron and multilayer perceptron implementation
Lucas Pires Camargo, 2022
Tópicos Especiais em Sistemas Eletrônicos IV - Aprendizado de Máquina
Programa de Pós-Graduação em Engenharia de Sistemas Eletrônicos – PPGESE
"""

# Model validation stuff goes here
# TODO unused and unfinished
# the idea was to implement k-cross at least
# maybe someday

from abc import ABC

import numpy as np


class ValidationMethod(ABC):
    # I guess the validation method itself would train the classifier(s)
    pass


# I'd have to rename all the 'partitiong' stuff and move it here (wrong name btw)
#class HoldOutValidation(ValidationMethod):
#    
#    def __init__(self, *params):
#        self.alpha = float(params[0])
#        assert 0.0 < self.alpha < 1.0

class KCrossValidation(ValidationMethod):
    
    def __init__(self, *params):
        self.n = int(params[0])


