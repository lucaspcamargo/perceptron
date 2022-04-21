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

from enum import Enum, unique
from math import floor
from random import shuffle
import numpy as np
from collections import namedtuple
from matplotlib import pyplot as plt

PROG_NAME = 'perceptron'
DEFAULT_TRAINING_EPOCHS_MAX = 1000 
DEFAULT_INITIAL_ETTA = 0.1      # Desired value of etta at the beginning of the training
DEFAULT_FINAL_ETTA = 0.00001    # Desired value of etta at the end of the training
DEFAULT_ETTA_GAMMA = 2.2        # Control etta shape with an exponent
DEFAULT_TRAINING_BATCH_SIZE = -1 # -1 means the entire dataset


class Instance:
    """
    An object of study in a Dataset
    """
    def __init__(self, klass, values):
        self._class = klass
        self.values = values

@unique
class PartitioningMode(Enum):
    NO_PARTITIONING = 'none',
    HOLDOUT = 'holdout'

@unique
class PartitioningContext(Enum):
    TRAINING = 1,
    VALIDATION = 2

PartitioningConfig = namedtuple("PartitioningConfig", "mode training_fraction", defaults={
    "mode": PartitioningMode.NO_PARTITIONING,
    'training_fraction': 0.8
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
        
        # Handle instance ordering and partitioning (eg holdout)
        shuffle(self._instances)
        self._part = partitioning_config
        if self._part.mode == PartitioningMode.HOLDOUT:
            validation_idx = floor(self._part.training_fraction * len(self._instances))
            self._instances_train = self._instances[:validation_idx]
            self._instances_validate = self._instances[validation_idx:]
            if not self._instances_train or not self._instances_validate:
                raise ValueError(f'[dataset] {data_filename}: not enough instances to partition! something must be wrong')
        else:
            # assuming no partitioning
            print(f'[dataset] {data_filename}: not using any partitioning, beware of overfitting perhaps?')
            self._instances_train = self._instances_validate = self._instances

    @property
    def params(self):
        return self._cols

    @property
    def classes(self):
        return self._classes

    @property
    def instances(self):
        return self._instances

    def instances_for(self, ctx:PartitioningContext = None):
        if not ctx:
            return self._instances
        if ctx == PartitioningContext.VALIDATION:
            return self._instances_validate
        elif ctx == PartitioningContext.TRAINING:
            return self._instances_train
        else:
            raise ValueError(f'[dataset] instances_for: unknown partitioning context')

    def scrambled_view(self, ctx:PartitioningContext = None):
        ret = list(self.instances_for(ctx))
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
        #print("[perceptron] init:",self._w)

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
                output_vec = perceptron.weights * input
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
        for i in domain_dataset.scrambled_view(PartitioningContext.TRAINING):
            self.train_iteration((i,), etta)
            #break # HACK FOR SIMPLE DATASET TODO param: batch size
 
    def train(self, domain_dataset:Dataset):
        etta_gamma = DEFAULT_ETTA_GAMMA
        etta_initial = pow(DEFAULT_INITIAL_ETTA, 1.0/etta_gamma)
        etta_final = pow(DEFAULT_FINAL_ETTA, 1.0/etta_gamma)
        max_epochs = DEFAULT_TRAINING_EPOCHS_MAX

        data_etta = []
        data_fraction = []

        for i in range(max_epochs):
            alpha = i/(max_epochs-1.0) # zero to one
            delta = 1.0-alpha # one to zero
            etta_curr = pow(etta_final + (etta_initial-etta_final)*delta, etta_gamma)
            self.train_epoch(domain_dataset, etta_curr)
            perfect, fraction = self.classify(domain_dataset, PartitioningContext.VALIDATION)
            #print(fraction*100, '%   rate=',etta_curr)
            data_etta.append(etta_curr)
            data_fraction.append(fraction*100)
            if perfect:
                print("[perceptron_classifier] train: all classification accurate, training complete")
                break
        else:
            print("[perceptron_classifier] train: no convergence")

        figure, axis = plt.subplots(2)
        axis[0].plot(range(len(data_etta)), data_etta)
        axis[0].set_title("Etta")
        axis[1].plot(range(len(data_fraction)), data_fraction)
        axis[1].set_title("Fraction")
        plt.show()

    
    def classify(self, dataset:Dataset, ctx:PartitioningContext=None):
        total = 0
        ok = 0
        nok_count = 0
        nok_wrong = 0
        item:Instance
        for item in dataset.instances_for(ctx):
            actual = item._class
            all_scores = []
            for klass, perceptron in self._p.items():
                input = np.concatenate((item.values, [self._bias,],))
                output_vec = perceptron.weights * input
                all_scores.append(self._activation(sum(output_vec)))
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

        return ok==total, float(ok)/total


def get_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(PROG_NAME)
    parser.add_argument('parser_name', help='id of the parser to use', type=str, nargs=1)
    parser.add_argument('dataset_dir', help='folder containing the dataset to use', type=str, nargs=1)
    parser.add_argument('dataset_info', help='name of the dataset in the folder, or additional dataset parser params', nargs="*", type=str)

    parser.add_argument('--max-epochs', '-e', help='maximum number of training epochs', nargs=1, type=int)
    parser.add_argument('--batch-size', '-b', help='number of epoch instances to train in a batch', nargs=1, type=int)
    return parser

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    print(args)

    import sys
    input_params = sys.argv[1]
    input_data = sys.argv[2]
    dataset = Dataset(input_params, input_data)
    classifier = PerceptronClassifier(dataset)
    print(classifier)

    print("Starting training")
    classifier.train(dataset)
    classifier.classify(dataset._instances_validate)
    

if __name__ == "__main__":
    main()
