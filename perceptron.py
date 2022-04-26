"""
perceptron and multilayer perceptron implementation
Lucas Pires Camargo, 2022
Tópicos Especiais em Sistemas Eletrônicos IV - Aprendizado de Máquina
Programa de Pós-Graduação em Engenharia de Sistemas Eletrônicos – PPGESE

- This tool implements Perceptron and Multi-Layer Perceptron classifiers,
in a variety of configurations.

- TODO list:
    - Numpy aready takes care of paralellization if you use the matrices right. CHECK
      Use it instead of mp.Pool. For that:
        - Consolidate dataset on a single matrix (add "bias" parameter automatically, set as one); CHECK
        - Consolidate all perceptrons of a layer on a single matrix (add "bias" random weight automatically); CHECK
        - Calculate the weighted inputs to all perceptrons on a layer at once, and then sum them up for every perceptron; CHECK
        - Apply activation function for the entire result using numpy stuff. CHECK
        - Calculate error using numpy. CHECK
        - Implement some actual batch handling.
        - Make all process arguments work.
    - Do MLP right the first time, taking advantage of the above; CHECK(?)
    - TODO review math
    - Allow storage of arbitrary per-epoch and per-batch stats, and plotting them;
    - Support different operations;
        - Classifier training and network save to file; - CHECK
        - Classifier load from file and classify dataset; - CHECK
    - Figure out what to do with partitioning;
    - Take a look at mROC;
    - Allow doing multiple training runs by varying the training parameters, and plotting them as multiple lines on the same graph for comparison;
    - BONUS: small UI tool that lets the user load/draw image to feed the neural network with, and see the outputs and some data - CHECK

- We are able to use the following datasets:
    - synthetic, linearly-separable (mine);
    - iris;
    - simplified handwritten digit dataset, from moodle;
    - MNIST.

- There is also a reference implementation that uses the scikit.learn implementation for performance comparison.
"""

# deps
from itertools import count
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

# comparison
from sklearn.neural_network import MLPClassifier as SKLMLPClassifier


##################################################################################################
# DEFAULTS

PROG_NAME = 'perceptron'
DEFAULT_TRAINING_EPOCHS_MAX = 1000 
DEFAULT_INITIAL_ETTA = 0.001      # Desired value of etta at the beginning of the training
DEFAULT_FINAL_ETTA = DEFAULT_INITIAL_ETTA     # Desired value of etta at the end of the training
DEFAULT_ETTA_GAMMA = 2.2         # Control etta shape with an exponent
DEFAULT_TRAINING_BATCH_SIZE = -1 # -1 means the entire dataset
DEFAULT_ERROR_THRESHOLD = 0.003
DEFAULT_MOMENTUM = 0.8
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
        return _cols, _instances, _classes, None

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
        return params, instances, classes, (8,8,)

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
        return params, instances, classes, (cols, rows,)

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

    def expected_output(self, classes):
         ret = np.zeros(len(classes))
         ret[classes.index(self._class)] = 1.0
         return ret

class Dataset:
    def __init__(self, source, partitioning_config=PartitioningConfig()):
        
        self._source = source
        self._cols, self._instances, self._classes, self.image_dims = source.read()
        print("Coalescing dataset (1/2)")
        self.param_mat = np.asarray([np.concatenate((i.values, [1.0,],)) for i in self._instances ])
        print("Coalescing dataset (2/2)")
        self.ref_mat = np.asarray([i.expected_output(self._classes) for i in self._instances ])
        print("Serializing dataset")
        self.param_mat.dump("dataset.dump")

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
        assert pcount
        assert icount
        self.weights = 2.0*np.random.random((icount, pcount,)) - np.ones((icount, pcount,))
        print(f"[perceptron_layer] init: {icount-1}+1 weights for {pcount} perceptrons")



class PerceptronClassifier:
    """
    Classification algorithm using a single perceptron per class.
    For this to work 100%, problem must be linearly separable.

    TODO DELETE THIS AFTER MLP works for this case and after plotting code is moved out and made better
    """
    def __init__(self, domain_dataset):
        self._p = {klass: Perceptron(domain_dataset) for klass in domain_dataset.classes}
        self._bias = 1.0
        self._activation = lambda x: 1.0 if x>=0.0 else 0.0  # TODO these values are literals in MP code
        self._match_val = self._activation(1.0)
        self._not_match_val = self._activation(-1.0)

    def __str__(self):
        return f'(PerceptronClassifier: {len(self._p)} perceptrons/classes)'

    @staticmethod
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
            err, frac = PerceptronClassifier.train_batch_static(self._p, b, etta, dry_run)
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



class SciKitLearnMLPClassifier:

    def __init__(self, dataset, args):
        self.dataset = dataset
        self._classes = dataset.classes
        self.args = args
        self.classifier = SKLMLPClassifier(
            solver='sgd',
            hidden_layer_sizes=tuple([int(x) for x in args.layout.split(',')]), 
            activation='logistic',
            batch_size=int(args.batch_size) if args.batch_size != -1 else 'auto',
            verbose=True,
            max_iter=int(args.max_epochs),
            learning_rate_init = DEFAULT_INITIAL_ETTA,
            learning_rate='constant',
            
            # ::::::TEST::::
            # TODO make this a config
            # make this work for MLP as well
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=100
        )
        

    def __str__(self):
        return f'(SciKitLearnMLPClassifier: {len(self.dataset.params)} inputs, {len(self.dataset.classes)} classes)'

    def train(self, domain_dataset:Dataset, args:argparse.Namespace):
        self.classifier.fit(domain_dataset.param_mat, domain_dataset.ref_mat)
        print(self.classifier)
        if args.plot:
            fig, axis = plt.subplots(2,1)
            plt.subplots_adjust(hspace = 0.4)
            axis[0].set_title("Loss curve")
            axis[0].plot(list(range(len(self.classifier.loss_curve_))), self.classifier.loss_curve_, c='purple')
            #axis[1].set_title("Intercepts")
            #axis[1].plot(list(range(len(self.classifier.intercepts_))), self.classifier.intercepts_, c='red')
            plt.show()
    
    def classify_single(self, single):
        single = np.atleast_2d(np.concatenate((single, [1.0,],)))
        res = self.classifier.predict(single)
        print(res)
        return res
        

class MLPClassifier:

    def __init__(self, domain_dataset:Dataset, args:argparse.Namespace):
        self._classes = domain_dataset.classes
        self._layers: list[PerceptronLayer] = []
        self._activation = MLPClassifier.sigmoid

        # build layers
        curr_input_sz = len(domain_dataset.params)+1
        hidden_szs = [int(x) for x in args.layout.split(',')] if args.layout else []
        for sz in hidden_szs:
            self._layers.append(PerceptronLayer(sz, curr_input_sz))
            curr_input_sz = sz+1 # +1 for bias

        # build output layer
        self._layers.append(PerceptronLayer(len(self._classes), curr_input_sz))

    def __str__(self):
        return f'(MLPClassifier: {self._layers[0].weights.shape[0]} inputs, {len(self._classes)} classes, {len(self._layers)} layers)'

    def _config_threads(self, args):
        import os
        if args.num_procs != -1:
            numstr = str(args.num_procs)
            for envvar in SMP_ENV_VARS:
                os.environ[envvar] = numstr

    @staticmethod
    def sigmoid(x:np.ndarray, derivative=False) -> np.ndarray:
        if derivative:
            return x*(1-x)
        return 1/(1+np.exp(-x))

    @staticmethod
    def step(x:np.ndarray, derivative=False) -> np.ndarray:
        # not to self: here heaviside is a sign() function with a special case for zero
        if derivative:
            return MLPClassifier.sigmoid(x) # use sigmoid to fake a derivative here
        return np.heaviside(x, 1.0 if not derivative else 0.0)
    
    def train(self, domain_dataset:Dataset, args:argparse.Namespace):

        self._config_threads(args)

        etta = DEFAULT_INITIAL_ETTA
        alpha_momentum = args.momentum
        params = domain_dataset.param_mat
        reference = domain_dataset.ref_mat

        print('MLP training::')
        print(f'classes:{len(domain_dataset.classes)} instances:{len(domain_dataset.instances)}')
        print(f'params:{params.shape} references:{reference.shape} batch_size:{args.batch_size}')

        dp_avg_cost = []
        dp_misclassify_count = []
    
        print(f"Go.")
        for i in range(args.max_epochs):
            try:
                #scramble params and reference matrices the same way
                assert len(params) == len(reference)
                perm = np.random.permutation(len(params))
                params = params[perm] # this will create copies but keeps the original matrices
                reference = reference[perm] # ditto

                # split the source in sets of "x" batches
                batches = [params[x:x+args.batch_size,:] for x in range(0, len(params), args.batch_size)] if args.batch_size != -1 else [params]
                refs = [reference[x:x+args.batch_size,:] for x in range(0, len(reference), args.batch_size)] if args.batch_size != -1 else [reference]

                epoch_misclassify_count_accum = []
                epoch_cost_accum = []
                momentum_prev = None

                batch:np.ndarray
                for batchnum, batch in enumerate(batches):
                    #print(f"Batch #{batchnum}, size {batchsize}")
                    ref = refs[batchnum]
                    ref_hit = np.greater(ref, 0.5) # TODO change to astype(bool) or something?
                    inputs = []
                    activations = []
                    outputs = []
                    deltas = []


                    # first, feed-forward
                    for layeridx, layer in enumerate(self._layers):
                        #print(f"FF layer {layeridx}")
                        layer_input = outputs[-1] if outputs else batch
                        if layer_input.shape[1] == layer.weights.shape[0]-1:
                            # bias missing in params
                            layer_input = np.concatenate((layer_input, np.ones((layer_input.shape[0],1,),),), 1)
                        #print(f"layer_input:{layer_input.shape} weights:{layer.weights.shape}")
                        inputs.append(layer_input)
                        activation = np.matmul(layer_input, layer.weights)
                        activations.append(activation)
                        output = self._activation(activation)
                        outputs.append(output)
                        if layer == self._layers[-1]:
                            # output layer, calc hit rate and delta on last step                    
                            # first, hit rate
                            output_hit = np.greater(output, 0.5)
                            misses = ref_hit^output_hit
                            sum_misses = np.sum(misses)
                            epoch_misclassify_count_accum.append(sum_misses)
                            # then, error -> loss -> cost
                            err = output - ref
                            err_sq = 0.5 * np.square(err)
                            losses = np.sum(err_sq, 1)/layer_input.shape[1] # cost per inst
                            #print(losses)
                            cost = np.sum(losses)
                            epoch_cost_accum.append(cost)
                            #print(f"calculating delta {layeridx} (output layer)")
                            #print(f'last_layer: layer_input:{layer_input.shape} cost:{cost}:avg={cost}')
                            delta = err*self._activation(output, True) # assuming sigmoid activation
                            deltas.append(delta)
                            #print(f'last_layer: w:{layer.weights.shape} etta:{etta} delta:{delta.shape}')
                        
                    # calculate other deltas backwards
                    for layeridx, layer in reversed(list(enumerate(self._layers[:-1]))):
                        #print(f"calculating delta {layeridx}")
                        delta_next = deltas[0]
                        if len(deltas) != 1:
                            delta_next = delta_next[:,:-1] # not an output delta, discard bias column
                        this_output = outputs[layeridx]
                        this_output = np.concatenate((this_output, np.ones((this_output.shape[0],1,),),), 1)
                        weights_n_t = self._layers[layeridx+1].weights.transpose()
                        #print(f"delta_next:{delta_next.shape} this_output:{this_output.shape} weights_n_t:{weights_n_t.shape}")
                        delta = np.matmul(delta_next, weights_n_t)*self._activation(this_output, True)
                        #print(f"delta:{delta.shape}")
                        deltas.insert(0, delta) # put first

                    assert len(inputs) == len(outputs) == len(deltas)    
                    
                    # now backpropagate
                    if not args.skip_dry_run:
                        new_momentum = []
                        for layeridx, layer in enumerate(self._layers):
                            gradient = np.matmul(inputs[layeridx].transpose(), deltas[layeridx])
                            if False:
                                print(f"adjusting {layeridx}")
                                print("inputs",inputs[layeridx].shape)
                                print("deltas",deltas[layeridx].shape)
                                print("weights",layer.weights.shape)
                                print("gradient",gradient.shape)
                            if gradient.shape[1] == layer.weights.shape[1]+1:
                                gradient = gradient[:,:-1] # TODO discard bias gradient? is this correct?
                            change = etta * gradient
                            if momentum_prev:
                                change += alpha_momentum*momentum_prev[layeridx]
                            new_momentum.append(change)
                            layer.weights -= change
                    momentum_prev = new_momentum

                    dump_batch_data = False
                    if dump_batch_data:
                        print("BATCH", batch)
                        print("WEIGHTS", layer.weights)
                        print("ACTIVATION", activation)
                        print("OUTPUT", output)
                        print("REF", ref)
                        print("ERROR", err)
                        print("ERRORSQ", err_sq)
                        print("COST", cost)
                        print("DELTA", delta)
                        print("LAST GRADIENT", gradient)
                
                # all batches done, account for the epoch:
                sum_misses = np.sum(epoch_misclassify_count_accum)
                dp_misclassify_count.append(sum_misses)
                cost = np.average(epoch_cost_accum)
                dp_avg_cost.append(cost)
                #print(weights.shape, outputs_float.shape, ref.shape, err.shape, err_squared.shape, learn.shape)
                # print(excitation)
                print(f"Epoch #{i}: end, cost={cost}, sum_misses={sum_misses}")
                if cost < 0.001:
                    break
            except KeyboardInterrupt:
                print('Trainign interrupted')
                break
               
        
        if args.plot:
            fig, axis = plt.subplots(2,1)
            plt.subplots_adjust(hspace = 0.4)
            axis[0].set_title("Average cost")
            axis[0].plot(list(range(len(dp_avg_cost))), dp_avg_cost, c='purple')
            axis[1].set_title("Misclassification count")
            axis[1].plot(list(range(len(dp_misclassify_count))), dp_misclassify_count, c='red')
            plt.show()

    def classify_single(self, single):
        """
        Classify a single input without bias.
        Parameter:
            - A list of input values, without bias column.
        """
        single = np.atleast_2d(np.concatenate((single, [1.0,],)))
        outputs = []
        for layeridx, layer in enumerate(self._layers):
            layer_input = outputs[-1] if outputs else single
            print(f"layer_input:{layer_input.shape} weights:{layer.weights.shape}")
            outputs_float = self.get_weighted_inputs(layeridx, layer_input)
            outputs_act = self._activation(outputs_float)
            outputs.append(outputs_act)
        return outputs[-1] # TODO deduplicate this feed-forward code

    
    def get_weighted_inputs(self, layeridx, params):
        w = self._layers[layeridx].weights
        if params.shape[1] == w.shape[0]-1:
            # bias missing in params
            params = np.concatenate((params, np.ones((params.shape[0],1,),),), 1)
        return np.matmul(params, w)

    def save(self, fname):
        weights = [l.weights for l in self._layers]
        np.savez_compressed(fname, *weights)

    def load(self, fname):
        loaded = np.load(fname)
        print(type(loaded))
        d = dict(loaded)
        for i, (k,v) in enumerate(d.items()):
            print(f"Loaded array named '{k}' into layer {i}")
            layer = self._layers[i]
            prev_shape = layer.weights.shape
            layer.weights = v
            curr_shape = layer.weights.shape
            assert prev_shape == curr_shape
            
        
        



##################################################################################################
# DATA COLLECTION

# ... todo graphs and stuff here


##################################################################################################
# TOOL

import sys
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtGui import QImage, QPainter, QPen, QKeyEvent, QFont


class DrawingTool(QMainWindow):
    """
    A window that allows the user to draw something and see how the classifier responds
    This was (stolen and) adapted from: https://stackoverflow.com/a/51475353
    """

    def __init__(self, classifier:MLPClassifier, img_dims):
        super().__init__()
        self.setWindowTitle(f"DrawingTool -- {classifier}")
        self.classifier = classifier
        self.drawing = False
        self.lastPoint = QPoint()
        self.image:QImage = QImage(img_dims[0], img_dims[1], QImage.Format.Format_Grayscale8)
        self.image.fill(Qt.black)
        self.scale = 20
        self.bottom_h = 20
        self.resize(self.image.width()*self.scale, self.image.height()*self.scale+self.bottom_h)
        self.draw_instructions = True
        self.reclassify()
        self.show()

    def keyPressEvent(self, a0: QKeyEvent) -> None:
        if a0.text().lower() == 'c':
            self.draw_instructions = True
            self.image.fill(Qt.black)
            self.reclassify()
            self.update()
            return
        elif a0.text().lower() == 'l':
            fname = QFileDialog.getOpenFileName(self, "Open an image")
            if not fname:
                return
            loaded = QImage(fname[0])
            self.image = loaded.scaled(self.image.width(), self.image.height())
            self.image.convertTo(QImage.Format.Format_Grayscale8)
            self.reclassify()
            self.update()
            return
        elif a0.text().lower() == 'r':
            fname = QFileDialog.getOpenFileName(self, "Open an image")
            if not fname:
                return
            loaded = QImage(fname[0])
            self.image = loaded.scaled(self.image.width(), self.image.height())
            self.image.convertTo(QImage.Format.Format_Grayscale8)
            self.reclassify()
            self.update()
            return
        return super().keyPressEvent(a0)

    def paintEvent(self, event):
        painter = QPainter(self)
        myrect = self.rect()
        myrect.setHeight(myrect.height()-self.bottom_h)
        painter.drawImage(myrect, self.image)
        painter.setFont(QFont("Monospace,mono,serif", 12, [-1, QFont.Bold][0], True))
        painter.fillRect(0,self.width(),self.width(),self.bottom_h,Qt.lightGray)
        txtpoint = QPoint(10,self.width()+15)
        painter.setPen(self.caption_color)
        painter.drawText(txtpoint, self.caption)
        if self.draw_instructions:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignVCenter) + int(Qt.AlignmentFlag.AlignHCenter), "Draw a number with the mouse\npress 'l' to open an image\npress 'r' to load a random object from the dataset")


    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()/self.scale

    def mouseMoveEvent(self, event):
        self.draw_instructions = False
        if event.buttons() and Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.white, 1, Qt.SolidLine))
            currpos = event.pos()/self.scale
            painter.drawLine(self.lastPoint, currpos)
            self.lastPoint = currpos
            self.reclassify()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False

    def reclassify(self):
        input = self.image.convertToFormat(QImage.Format.Format_Grayscale8)
        ptr = input.constBits()
        ptr.setsize(input.height() * input.width())
        arr:np.ndarray = np.frombuffer(ptr, np.uint8) 
        arrfloat = arr.astype(np.float32)
        print(arrfloat, arrfloat.shape)
        out = self.classifier.classify_single(arrfloat)
        print(out)

        tbl = ""
        ctbl = ""
        for i,c in enumerate(self.classifier._classes):
            val = out[0][i]
            prefix = "\u001b[33m\u001b[1m" if val else ""
            suffix = '\u001b[0m' if val else ''
            ctbl+=f'{prefix}{c}:{val}{suffix}\t'
            if val > 0.66:
                tbl+=f'{c}:{val} '
        self.caption = tbl if tbl else "(no classes match)"
        if not self.draw_instructions:
            self.caption += " - press 'c' to clear"
        self.caption_color = (Qt.blue if tbl.count(":") == 1 else Qt.black) if tbl else Qt.red
        print(tbl)

    @staticmethod
    def run(classifier, image_dims):
        app = QApplication(sys.argv)
        main = DrawingTool(classifier, image_dims)
        (main) # unused ref
        sys.exit(app.exec_())


def plot_confusion(classifier):
    pass


def get_arg_parser():
    parser = argparse.ArgumentParser(PROG_NAME)
    parser.add_argument('parser_name', help='id of the parser to use', type=str, nargs='?')
    parser.add_argument('dataset_info', help='required dataset filenames, or additional dataset parser parameters', nargs="*", type=str)

    parser.add_argument('--layout', '-l', help='Comma-separated list of hidden layer sizes. Defaults to no hidden layers (simple perceptron).', 
                        nargs='?', type=str, default="")
    parser.add_argument('--max-epochs', '-e', help='maximum number of training epochs', nargs='?', type=int, default=DEFAULT_TRAINING_EPOCHS_MAX)
    parser.add_argument('--batch-size', '-b', help='number of epoch instances to train in a batch', nargs='?', type=int, default=DEFAULT_TRAINING_BATCH_SIZE)
    parser.add_argument('--num-procs', '-j', help='number of processes used for training', nargs='?', type=int, default=DEFAULT_TRAINING_NUM_PROCS)
    parser.add_argument('--error-threshold', '-t', help='Error level to consider convergence', nargs='?', type=float, default=DEFAULT_ERROR_THRESHOLD)
    parser.add_argument('--momentum', '-m', help=f'Training momentum (default is {DEFAULT_MOMENTUM})', nargs='?', type=float, default=DEFAULT_MOMENTUM)
    parser.add_argument('--skip-dry-run', '-s', help='Skip making epoch #0 a dry-run', action='store_true')
    parser.add_argument('--plot', '-p', help='Show plots after training', action='store_true')
    parser.add_argument('--draw', '-d', help='Launch drawing tool to test classifier', action='store_true')
    parser.add_argument('--reference', '-r', help='Use reference classifier', action='store_true')
    parser.add_argument('--save', '-S', help='Save model to file aft3er training', nargs='?', type=str)
    parser.add_argument('--load', '-L', help='Load model from file aft3er training', nargs='?', type=str)
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

    datasource = get_datasource(args)
    dataset = Dataset(datasource)
    classifier = (MLPClassifier if not args.reference else SciKitLearnMLPClassifier)(dataset, args)
    print(classifier)

    assert not (args.save and args.load)

    if args.load:
        print("Loading training")
        classifier.load(args.load)
    else:
        print("Starting training")
        classifier.train(dataset, args)


    if args.save:
        print(f"Saving model to {args.save}")
        classifier.save(args.save)

    #print("Final classification...")
    #print(classifier.classify(dataset))

    if args.draw:
        if dataset.image_dims:
            print("Drawing tool...")
            DrawingTool.run(classifier, dataset.image_dims)
        else:
            print("Dataset has no image dimensions for input. Cannot run drawing tool.")
    

if __name__ == "__main__":
    main()
