"""
perceptron and multilayer perceptron implementation
Lucas Pires Camargo, 2022
Tópicos Especiais em Sistemas Eletrônicos IV - Aprendizado de Máquina
Programa de Pós-Graduação em Engenharia de Sistemas Eletrônicos – PPGESE

- This tool implements Perceptron and Multi-Layer Perceptron classifiers,
in a variety of configurations.

- TODO list:
    - Numpy aready takes care of paralellization if you use the matrices right.
      Use it instead of mp.Pool. For that:
        - Consolidate dataset on a single matrix (add "bias" parameter automatically, set as one);
        - Consolidate all perceptrons of a layer on a single matrix (add "bias" random weight automatically);
        - Calculate the weighted inputs to all perceptrons on a layer at once, and then sum them up for every perceptron;
        - Apply activation function for the entire result using numpy stuff.
        - Calculate error using numpy.
        - By this point, you know.
    - Do MLP right the first time, taking advantage of the above;
    - Support different operations (with argparser subcommands maybe);
        - Classifier training and network save to file;
        - Classifier load from file and classify dataset;
    - Figure out what to do with partitioning;
    - Take a look at mROC;
    - Allow storage of arbitrary per-epoch and per-batch stats, and plotting them;
    - Allow doing multiple training runs by varying the training parameters, and plotting them as multiple lines on the same graph for comparison;
    - BONUS: small UI tool that lets the user load/draw image to feed the neural network with, and see the outputs and some data.

- We are able to use the following datasets:
    - synthetic, linearly-separable (mine);
    - iris;
    - simplified handwritten digit dataset, from moodle;
    - MNIST.
"""

# deps
import numpy as np
from matplotlib import pyplot as plt

# std
from abc import ABC
import argparse
from enum import Enum, unique
from math import floor
from os import getpid
from random import shuffle
from collections import namedtuple
from PIL import Image
import multiprocessing as mp


##################################################################################################
# DEFAULTS

PROG_NAME = 'perceptron'
DEFAULT_TRAINING_EPOCHS_MAX = 1000 
DEFAULT_INITIAL_ETTA = 0.01      # Desired value of etta at the beginning of the training
DEFAULT_FINAL_ETTA = DEFAULT_INITIAL_ETTA     # Desired value of etta at the end of the training
DEFAULT_ETTA_GAMMA = 2.2         # Control etta shape with an exponent
DEFAULT_TRAINING_BATCH_SIZE = -1 # -1 means the entire dataset
DEFAULT_ERROR_THRESHOLD = 0.001
DEFAULT_TRAINING_NUM_PROCS = 1
# DEBUG_FLAGS
DUMP_ITERATION = False


##################################################################################################
# DEFINITIONS

SMP_ENV_VARS = [
    "OMP_ENV_VARS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]

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

##################################################################################################
# DATASET SOURCES

class DatasetSource(ABC):

    def read(self):
        """Returns two things, a list of instances and class names"""
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

class IrisDatasetSource(DatasetSource):

    def __init__(self, params_filename, data_filename):
        self.params_filename = params_filename
        self.data_filename = data_filename

    def read(self):
        _instances = []
        with open(self.params_filename, 'r') as fcols:
            lines = fcols.readlines()
            stripped = [x.strip() for x in lines if x.strip()]
            assert stripped[-1] == 'class'
            _cols = stripped[:-1]
        print(f'[dataset] {self.params_filename}: loaded params: {_cols}')
        _classes = list()
        num_params = len(_cols)
        with open(self.data_filename, 'r') as fdset:
            lines = fdset.readlines()
            stripped = [x.strip() for x in lines if x.strip()]
            for line in stripped:
                tokens = line.split(',')
                values = [float(x) for x in tokens[:-1]]
                assert len(values) == num_params
                klass = tokens[-1]    
                if(klass not in _classes):
                    _classes.append(klass)
                instance = Instance(klass, np.array(values))
                _instances.append(instance)
        print(f'[dataset] {self.data_filename}: loaded {len(_instances)} instances')
        return _cols, _instances, _classes

    def __str__(self):
        return f'IrisDatasetSource("{self.params_filename}","{self.data_filename}")'

class MoodleDatasetSource(DatasetSource):
    def __init__(self, filename, *flags) -> None:
        super().__init__()
        self.filename = filename
        self.flags = flags

    def read(self):
        instances = []
        dump = "dump" in self.flags

        with open(self.filename, 'r') as f:
            lines = f.readlines()
            stripped = [x.strip() for x in lines if x.strip()]
            idx = 0
            for line in stripped:
                tokens = line.split(',')
                klass = str(tokens[0])
                values = [float(x) for x in tokens[1:]]
                instance = Instance(klass, np.array(values))
                print(instance)
                instances.append(instance)
                idx += 1
                if dump:
                    valuesi = [int(x) for x in tokens[1:]]
                    reshaped = np.array(valuesi).reshape(8,8)
                    as_bytes = (255*reshaped).astype(np.uint8)
                    im:Image.Image = Image.fromarray(as_bytes, 'L')
                    im.save(f"./{idx}.png")

        params = [f'p{x}' for x in range(64)]
        classes = [str(x) for x in range(10)]
        print(params)
        print(classes)
        return params, instances, classes

    def __str__(self):
        return f'MoodleDatasetSource("{self.filename}")'

class MNISTDatasetSource(DatasetSource):
    def __init__(self, fname_labels, fname_images, *flags):
        self.fname_labels = fname_labels
        self.fname_images = fname_images
        self.flags = flags

    def read(self):
        dump = "dump" in self.flags
        instances = []

        with open(self.fname_labels, 'rb') as flbl:
            magic = flbl.read(4)
            assert magic==b'\0\0\x08\x01'
            countbytes = flbl.read(4)
            count = int.from_bytes(countbytes, byteorder='big')
            labels = [int(b) for b in flbl.read(count)]

        with open(self.fname_images, 'rb') as flbl:
            header = flbl.read(4*4)
            magic = header[0:4]
            assert magic==b'\0\0\x08\x03'
            count = int.from_bytes(header[4:8], byteorder='big')
            rows = int.from_bytes(header[8:12], byteorder='big')
            cols = int.from_bytes(header[12:16], byteorder='big')
            print(f"[mnist] importing {count} images with {rows}x{cols} pixels each")
            imsz = rows*cols
            for i in range(count):
                pixels = flbl.read(imsz)
                instance_data = np.frombuffer(pixels, dtype=np.uint8).astype(float)/255.0
                inst = Instance(labels[i], instance_data)
                instances.append(inst)
                if dump:
                    valuesi = np.frombuffer(pixels, dtype=np.uint8)
                    reshaped = np.array(valuesi).reshape(rows,cols)
                    im:Image.Image = Image.fromarray(reshaped, 'L')
                    im.save(f"./{i}.png")

        params = [f'p{x}' for x in range(imsz)]
        classes = list(set(labels))
        return params, instances, classes

    def __str__(self):
        return f'MoodleDatasetSource("{self.fname_images}","{self.fname_labels}")'

##################################################################################################
# MODELS


class Instance:
    """
    An object of study in a Dataset
    """
    def __init__(self, klass, values):
        self._class = klass
        self.values = values

    def __str__(self):
        return f"(Instance class={self._class}, params: {self.values})"

class Dataset:
    def __init__(self, source, partitioning_config=PartitioningConfig()):
        
        self._source = source
        self._cols, self._instances, self._classes = source.read()

        if not self._cols:
            raise ValueError("Dataset source has no parameter ids!")
        if not self._instances:
            raise ValueError("Dataset source has no instances!")
        if not self._classes:
            raise ValueError("Dataset source has no classes!")
        
        # Handle instance ordering and partitioning (eg holdout)
        shuffle(self._instances)
        self._part = partitioning_config
        if self._part.mode == PartitioningMode.HOLDOUT:
            validation_idx = floor(self._part.training_fraction * len(self._instances))
            self._instances_train = self._instances[:validation_idx]
            self._instances_validate = self._instances[validation_idx:]
            if not self._instances_train or not self._instances_validate:
                raise ValueError(f'[dataset] {self._source}: not enough instances to partition! something must be wrong')
        else:
            # assuming no partitioning
            print(f'[dataset] {self._source}: not using any partitioning, beware of overfitting perhaps?')
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


##################################################################################################
# CLASSIFIERS

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


class PerceptronLayer:

    def __init__(self, pcount, icount):
        """
        Create a Perceptron layer according to the dataset domain.
        Params:
            - pcount: number of perceptrons in the layer
            - icount: number of inputs for every perceptron (e.g. number of dataset parameters + 1 for the bias)
        """
        # plus 1 for bias
        assert(pcount)
        wlen = len(icount)
        self._w = 2.0*np.random.random((pcount, wlen,)) - np.ones((pcount, wlen,))
        print(f"[perceptron_layer] init: {wlen-1}+1 weights for {pcount} perceptrons",self._w)

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
        self._activation = lambda x: 1.0 if x>=0.0 else 0.0  # TODO these values are literals in MP code
        self._match_val = self._activation(1.0)
        self._not_match_val = self._activation(-1.0)

    def __str__(self):
        return f'(PerceptronClassifier: {len(self._p)} perceptrons/classes)'

    def train_batch(self, batch, etta, dry_run):
        item: Instance        
        errors = []
        fraction = []
        for item in batch:
            # train item
            for klass, perceptron in self._p.items():
                expected = self._match_val if klass==item._class else self._not_match_val
                input = np.concatenate((item.values, [self._bias,],)) # TODO simplify this, put a bias with all values automatically
                output_vec = perceptron.weights * input
                output = self._activation(sum(output_vec))
                err = expected - output
                learn = err * input
                final_weights =  perceptron.weights + learn*etta
                if DUMP_ITERATION:
                    print('--')
                    print("values", item.values)
                    print("weights", perceptron.weights)
                    print("output_vec", output_vec)
                    print("output", output)
                    print("expected", expected)
                    print("err", err)
                    print("learn", learn)
                    print("etta", etta)
                    print("learn*etta", learn*etta)
                    print("final_weights", final_weights)

                if not dry_run:
                    perceptron.weights = final_weights
                errors.append(err*err)
                fraction.append(0 if output!=expected else 1)
        return np.average(errors), np.average(fraction)


    def train_batch_static(perceptrons, batch, etta, dry_run):
        item: Instance        
        errors = []
        fraction = []
        for item in batch:
            # train item
            for klass, perceptron in perceptrons.items():
                expected = 1.0 if klass==item._class else 0.0
                input = np.concatenate((item.values, [1.0,],)) # TODO simplify this, put a bias with all values automatically
                output_vec = perceptron.weights * input
                output = 1.0 if (sum(output_vec)) >= 0.0 else 0.0
                err = expected - output
                learn = err * input
                final_weights =  perceptron.weights + learn*etta
                if not dry_run:
                    perceptron.weights = final_weights
                errors.append(err*err)
                fraction.append(0 if output!=expected else 1)
        return np.average(errors), np.average(fraction)


    def train_epoch(self, domain_dataset:Dataset, etta, args:argparse.Namespace, dry_run):

        source = domain_dataset.scrambled_view(PartitioningContext.TRAINING)
        # split the source in sets of "x" batches
        batches = [source[x:x+args.batch_size] for x in range(0, len(source), args.batch_size)] if args.batch_size != -1 else [source]
        
        size = []
        errors = []
        fraction = []

        #process all batches one after the other
        for i, b in enumerate(batches):
            if True: # use static func for single-thread too
                err, frac = PerceptronClassifier.train_batch_static(self._p, b, etta, dry_run)
            else:
                err, frac = self.train_batch(b, etta, dry_run)
            size.append(len(b))
            errors.append(err)
            fraction.append(frac)
            print(f"Batch #{i}: err={err}, fraction={frac}")
        #print(size, errors, fraction) 
        return np.average(sum([a*b for a,b in zip(size, errors)])/sum(size)),\
               np.average(sum([a*b for a,b in zip(size, fraction)])/sum(size))
 

    @staticmethod
    def train_batch_mp_main(work):
        perceptrons = work[0]
        batch = work[1]
        etta = work[2]
        dry_run = work[3]
        #print(f"train_batch_mp_main, pid={getpid()}")
        errs, frac = PerceptronClassifier.train_batch_static(perceptrons, batch, etta, dry_run)
        return len(batch), list(map(lambda k: (k, list(perceptrons[k].weights)), perceptrons.keys())), (errs, frac,)

    def train_epoch_mp(self, domain_dataset:Dataset, etta, args:argparse.Namespace, dry_run):
        source = domain_dataset.scrambled_view(PartitioningContext.TRAINING)
        # split the source in sets of "x" batches
        batches = [source[x:x+args.batch_size] for x in range(0, len(source), args.batch_size)] if args.batch_size != -1 else [source]
        
        size = []
        errors = []
        fraction = []

        num_procs = args.num_procs
        collected = 0
        while batches:
            slice = batches[:num_procs]
            batches = batches[num_procs:]
            pool = mp.Pool(num_procs)
            results = pool.map(PerceptronClassifier.train_batch_mp_main, [(self._p, batch, etta, dry_run) for batch in slice]) # THIS WILL NEVER WORK
            #print("MP RESULTS::: " + print(results))

            size_accum = 0
            weight_accum = {}

            #process batches in slice
            for i, bres in enumerate(results):
                batch_size = bres[0]
                new_weights = {k:v for k,v in bres[1]}
                err, frac = bres[2]
                print(f"Batch #{collected}: sz={batch_size}, err={err}, fraction={frac} (collecting)")
                # for every perceptron, accumulate the weights, weighted by this batches' size
                for k in new_weights:
                    weighted_weights = float(batch_size)*np.array(new_weights[k])
                    weight_accum[k] = weighted_weights if i==0 else (weight_accum[k]+weighted_weights)
                #print(bres)
                #print(new_weights[0])
                size.append(batch_size)
                size_accum += batch_size
                errors.append(err)
                fraction.append(frac)
                collected += 1

            # at this point, we got all results from current slice
            assert size_accum == sum(map(len,slice))

            # for every perceptron, update weights
            for k in self._p:
                final_weights = (1.0/float(size_accum)) * weight_accum[k]
                if not dry_run:
                    target = self._p[k]
                    assert final_weights.shape == target.weights.shape
                    target.weights = final_weights

        #print(size, errors, fraction)
        size_total = float(sum(size))
        return sum([a*b for a,b in zip(size, errors)])/size_total,\
               sum([a*b for a,b in zip(size, fraction)])/size_total


    def train(self, domain_dataset:Dataset, args:argparse.Namespace):
        etta_gamma = DEFAULT_ETTA_GAMMA
        etta_initial = pow(DEFAULT_INITIAL_ETTA, 1.0/etta_gamma)
        etta_final = pow(DEFAULT_FINAL_ETTA, 1.0/etta_gamma)
        max_epochs = DEFAULT_TRAINING_EPOCHS_MAX
        error_threshold = args.error_threshold

        data_etta = []
        data_error = []
        fig, axis = plt.subplots(2)

        for i in range(max_epochs):
            try:
                alpha = i/(max_epochs-1.0) # zero to one
                delta = 1.0-alpha # one to zero
                etta_curr = pow(etta_final + (etta_initial-etta_final)*delta, etta_gamma)
                error, fraction = (self.train_epoch if args.num_procs == 1 else self.train_epoch_mp)(domain_dataset, etta_curr, args, i==0 and not args.skip_dry_run)
                print(f"Epoch #{i}: err={error}, fraction={fraction}, rate={etta_curr}")
                data_etta.append(etta_curr)
                data_error.append(error)
                if data_etta and data_error:
                    axis[0].cla()
                    axis[0].set_title("Etta")
                    axis[0].plot(range(len(data_etta)), data_etta, c='blue')
                    axis[1].cla()
                    axis[1].set_title("Error")
                    axis[1].plot(range(len(data_error)), data_error, c='red')
                    try:
                        fig.canvas.draw()
                        plt.pause(0.05)
                    except KeyboardInterrupt:
                        raise
                if error <= error_threshold:
                    print(f"[perceptron_classifier] train: error threshold met ({error} <= {error_threshold})")
                    break
            except KeyboardInterrupt:
                print("[perceptron_classifier] training interrupted")
                break
        else:
            print("[perceptron_classifier] train: no convergence")

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



class MLPClassifier:

    def __init__(self, domain_dataset):
        self._p = {klass: Perceptron(domain_dataset) for klass in domain_dataset.classes}
        self._bias = 1.0
        self._activation = lambda x: 1.0 if x>=0.0 else 0.0  # TODO these values are literals in MP code
        self._match_val = self._activation(1.0)
        self._not_match_val = self._activation(-1.0)

    def _config_threads(self, args):
        import os
        if args.num_procs != -1:
            numstr = str(args.num_procs)
            for envvar in SMP_ENV_VARS:
                os.environ[envvar] = numstr
            





##################################################################################################
# DATA COLLECTION

# ... todo graphs and stuff here


##################################################################################################
# TOOL

def get_arg_parser():
    parser = argparse.ArgumentParser(PROG_NAME)
    parser.add_argument('parser_name', help='id of the parser to use', type=str, nargs='?')
    parser.add_argument('dataset_info', help='required dataset filenames, or additional dataset parser parameters', nargs="*", type=str)

    parser.add_argument('--max-epochs', '-e', help='maximum number of training epochs', nargs=1, type=int, default=DEFAULT_TRAINING_EPOCHS_MAX)
    parser.add_argument('--batch-size', '-b', help='number of epoch instances to train in a batch', nargs='?', type=int, default=DEFAULT_TRAINING_BATCH_SIZE)
    parser.add_argument('--num-procs', '-j', help='number of processes used for training', nargs='?', type=int, default=DEFAULT_TRAINING_NUM_PROCS)
    parser.add_argument('--error-threshold', '-t', help='Error level to consider convergence', nargs='?', type=float, default=DEFAULT_ERROR_THRESHOLD)
    parser.add_argument('--skip-dry-run', '-s', help='Skip making epoch #0 a dry-run', action='store_true')
    return parser

def get_datasource(args):
    if args.parser_name == "iris":
        return IrisDatasetSource(*args.dataset_info)
    if args.parser_name == "moodle":
        return MoodleDatasetSource(*args.dataset_info)
    if args.parser_name == "mnist":
        return MNISTDatasetSource(*args.dataset_info)
    else:
        raise ValueError(f"unknown parser: {args.parser_name}")

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    print(args)

    #datasource = IrisDatasetSource("iris/params", "iris/iris.data")
    #datasource = MoodleDatasetSource("moodle/data.csv")
    datasource = get_datasource(args)
    dataset = Dataset(datasource)
    classifier = PerceptronClassifier(dataset)
    print(classifier)

    print("Starting training")
    classifier.train(dataset, args)

    print("Final classification...")
    print(classifier.classify(dataset))
    

if __name__ == "__main__":
    main()
