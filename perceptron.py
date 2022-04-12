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
from typing import final
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
        self._classes = list()
        num_params = len(self.params)
        with open(data_filename, 'r') as fdset:
            lines = fdset.readlines()
            stripped = [x.strip() for x in lines if x.strip()]
            for line in stripped:
                tokens = line.split(',')
                values = [float(x) for x in tokens[:-1]]
                assert len(values) == num_params
                klass = tokens[-1]    
                if(klass not in self._classes):
                    self._classes.append(klass)
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
        # plus 1 for bias
        wlen =len(domain_dataset.params) + 1
        self._w = 2.0*np.random.random(wlen) - np.ones(wlen)
        print("[perceptron] init:",self._w)

    @property
    def weights(self):
        return self._w

    @weights.setter
    def weights(self, wval):
        assert(type(self._w)==type(wval))
        self._w = wval

    @property
    def weights_T(self):
        return np.atleast_2d(self._w).transpose()

    def __str__(self):
        return f'(Perceptron: {self._w})'



class PerceptronClassifier:
    """
    Classification algorithm using a single perceptron per class.
    For this to work 100%, problem must be linearly separable.
    """
    def __init__(self, domain_dataset):
        self._p = {klass: Perceptron(domain_dataset) for klass in domain_dataset.classes}
        self._bias = 1.0
        self._activation = lambda x: 1.0 if x>=0.0 else 0.0
        self._match_val = self._activation(1.0)
        self._not_match_val = self._activation(-1.0)

    def __str__(self):
        return f'(PerceptronClassifier: {self._p})'

    def train_iteration(self, batch, etta):
        item: Instance
        for item in batch:
            # train item
            for klass, perceptron in self._p.items():
                expected = self._match_val if klass==item._class else self._not_match_val
                input = np.concatenate((item.values, [self._bias,],))
                output_vec = input * perceptron.weights
                output = self._activation(sum(output_vec))
                err = expected - output
                learn = err * input
                final_weights =  perceptron.weights + learn*etta
                #print('--')
                #print("values", item.values)
                #print("weights", perceptron.weights)
                #print("output_vec", output_vec)
                #print("output", output)
                #print("expected", expected)
                #print("err", err)
                #print("learn", learn)
                #print("etta", etta)
                #print("learn*etta", learn*etta)
                #print("final_weights", final_weights)

                perceptron.weights = final_weights

    def train_epoch(self, domain_dataset:Dataset, etta):
        #print('=======EPOCH======')
        for i in domain_dataset.scrambled_view:
            self.train_iteration((i,), etta)
 
    def train(self, domain_dataset:Dataset):
        etta_initial = 1.0
        etta_final = 0.0
        num_epochs = 50000
        for i in range(num_epochs):
            alpha = i/(num_epochs-1.0)
            delta = 1.0-alpha
            #print(alpha, delta)
            self.train_epoch(domain_dataset, etta_final + (etta_initial-etta_final)*delta)

    
    def classify(self, dataset:Dataset):
        total = 0
        ok = 0
        nok_count = 0
        nok_wrong = 0
        item:Instance
        for item in dataset.instances:
            actual = item._class
            all_scores = []
            for klass, perceptron in self._p.items():
                input = np.concatenate((item.values, [self._bias,],))
                output_vec = input * perceptron.weights
                all_scores.append(self._activation(sum(output_vec)))
            print(all_scores)
            total += 1
            matches = all_scores.count(self._match_val)
            if matches == 1:
                if all_scores.index(self._match_val) == dataset.classes.index(item._class):
                    ok += 1
                    #print("match")
                else:
                    nok_wrong += 1
                    #print("matched_wrong", all_scores.index(self._match_val), dataset.classes.index(item._class), item._class )
            else:
                #print("bad")
                nok_count += 1

        print(self._p)
        print(dataset.classes)
        print(total, ok, nok_count, nok_wrong)

def main():
    import sys
    input_params = sys.argv[1]
    input_data = sys.argv[2]
    dataset = Dataset(input_params, input_data)
    classifier = PerceptronClassifier(dataset)
    print(classifier)

    print("Starting training")
    classifier.train(dataset)
    classifier.classify(dataset)
    

if __name__ == "__main__":
    main()
