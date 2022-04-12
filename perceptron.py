"""
perceptron and multilayer perceptron implementation
Lucas Pires Camargo, 2022
Tópicos Especiais em Sistemas Eletrônicos IV - Aprendizado de Máquina
Programa de Pós-Graduação em Engenharia de Sistemas Eletrônicos – PPGESE


Here are my self-imposed rules:
- Use numpy, in preparation for more computationally-intensive algorithms. Get used to the matrix jank.
- Performance is nice to have, but legibility is paramount.
- Terseness is neither encouraged nor frowned upon. But if the intent is not clear, I lose.
- Performance may suck or not. I want to use batched training, so that I can parallelize this
  with understandable semantics, and bring my Ryzen to its knees. We'll see.
People crap all over OO these days, but considering the above, we are doing this with classes. Fight me.

Other considerations:
- We are able to use the following datasets:
    - iris (for P);
    - MNIST (for (MLP). TODO 
"""

from enum import Enum
from random import shuffle
import numpy as np
from collections import namedtuple

IMG_W = 128
STEPS = 1024
ETTA_INITIAL = 1.0

# teste simples com problema linearmente separavel
# base de dados inicial: iris dataset: https://archive.ics.uci.edu/ml/datasets/iris


class Instance:
    def __init__(self, klass, values):
        self._class = klass
        self.values = values

class PartitioningMode(Enum):
    NO_PARTITIONING = 0,
    TRAINING_AND_VALIDATION = 1

PartitioningConfig = namedtuple("PartitioningConfig", "mode training_fraction", defaults={
    "mode": PartitioningMode.NO_PARTITIONING,
    'training_fraction': 0.75
})

class Dataset:
    def __init__(self, params_filename, data_filename, partitioning_config=PartitioningConfig()):
        self._instances = []
        with open(params_filename, 'r') as fcols:
            lines = fcols.readlines()
            stripped = [x.strip() for x in lines if x.strip()]
            assert stripped[-1] == 'class'
            self._cols = stripped[:-1]
        print(f'[dataset] {params_filename}: loaded params: {self.params}')
        self._classes = set()
        num_params = len(self.params)
        with open(data_filename, 'r') as fdset:
            lines = fdset.readlines()
            stripped = [x.strip() for x in lines if x.strip()]
            for line in stripped:
                tokens = line.split(',')
                values = [float(x) for x in tokens[:-1]]
                assert len(values) == num_params
                klass = tokens[-1]    
                self._classes.add(klass)
                instance = Instance(klass, np.array(values))
                self._instances.append(instance)
        print(f'[dataset] {data_filename}: loaded {len(self.instances)} instances')
        self._part = partitioning_config

    @property
    def params(self):
        return self._cols


    @property
    def classes(self):
        return self._classes

    @property
    def instances(self):
        return self._instances

    @property
    def scrambled_view(self):
        """
        This returns 
        """
        ret = list(self.instances)
        shuffle(ret)
        return ret


class Perceptron:

    def __init__(self, domain_dataset):
        """
        Create a Perceptron according to the dataset domain.
        """
        self._w = np.random.random(len(domain_dataset.params))

    @property
    def weights(self):
        return self._w

    def __str__(self):
        return f'(Perceptron: {self._w})'


class PerceptronClassifier:
    """
    Classification algorithm using a single perceptron per class.
    For this to work 100%, problem must be 100% linearly separable.
    """
    def __init__(self, domain_dataset):
        self._p = {param: Perceptron(domain_dataset) for param in domain_dataset.params}

    def __str__(self):
        return f'(PerceptronClassifier: {self._p})'

    def train_batch_iteration(self):
        pass

    def train_epoch(self, domain_dataset):
        pass
 
    def train(self):
        pass


def main():
    import sys
    input_params = sys.argv[1]
    input_data = sys.argv[2]
    dataset = Dataset(input_params, input_data)
    classifier = PerceptronClassifier(dataset)
    print(classifier)
    

if __name__ == "__main__":
    main()


""""
w = np.zeros(IMG_W*IMG_W)
def etta(i:int):
    return ETTA_INITIAL

def sgn(x):
    return x #TODO

def training():

    for i in range(STEPS):

        y[i] = sgn(w[i].transposed() * x[i])

        # adaptacao de pesos:
        w[i+1] = w[i] + etta(i) * (desired[i] - y[i]) * x[i]

        # TODO: embaralhar os dados em épocas
        
        
def classify():
    pass
"""