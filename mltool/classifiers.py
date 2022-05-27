
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

from mltool.models import Dataset, Instance
from mltool.defs import (
    PartitioningContext,
    DEFAULT_TRAINING_EPOCHS_MAX,
    SMP_ENV_VARS,
    DEFAULT_INITIAL_ETTA
)
from mltool.plots import TrainingGraph



##################################################################################################
# PARAMS

TrainingParams = namedtuple("TrainingParams", "max_epochs batch_size layout etta momentum")
def parse_variable_param(val:str, num_t:type):
    if type(val) != str: 
        return val
    if ";" in val:
        return [num_t(x) for x in val.split(";")]
    elif ":" in val:
        if val.count(":") == 2:
            tokens = val.split(":")
            print(tokens)
            if num_t == int:
                ntokens = [int(x) for x in tokens]
                ret = list(range(ntokens[0], ntokens[2], ntokens[1]))
                ret.append(ntokens[2])
                return ret
            elif num_t == float:
                ftokens = [float(x) for x in tokens]
                return list(np.arange(ftokens[0], ftokens[2], ftokens[1])) + [float(ftokens[-1])]
            else:
                raise ValueError(f"[parse_variable_param] don't know how to interpolate type: {repr(num__t)}")
    else:
        return [num_t(val),]

def cross_training_params(tp:TrainingParams):
    import itertools
    print(list(tp))
    params = list(tp)
    params = [x if isinstance(x, list) else [x,] for x in params]
    all_combos = list(itertools.product(*params))
    print('all_combos::',all_combos)
    ret = {}
    for combo in all_combos:
        ret["_".join([str(x) for x in combo])] = combo
    return ret

def save_training_params(basename:str, tp:TrainingParams, combos:list):
    print(tp)
    import json
    with open(basename+".tp.json", "w") as fileobj:
        fileobj.write(json.dumps({
            "params": tp._asdict(),
            "combinations": combos
        }))



##################################################################################################
# CLASSIFIERS


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

    @staticmethod
    def batchsplit(brute:np.ndarray, sz:int):
        return [brute[x:x+sz,:] for x in range(0, len(brute), sz)]\
                            if sz != -1 else [brute]
    
    def train(self, domain_dataset:Dataset, args:argparse.Namespace, tg:TrainingGraph, tid:str):

        etta = args.etta
        alpha_momentum = args.momentum
        params = domain_dataset.param_mat_for(PartitioningContext.TRAINING)
        reference = domain_dataset.ref_mat_for(PartitioningContext.TRAINING)
        v_params = domain_dataset.param_mat_for(PartitioningContext.VALIDATION)
        v_reference = domain_dataset.ref_mat_for(PartitioningContext.VALIDATION)

        print('MLP training::')
        print(f'classes:{len(domain_dataset.classes)} instances:{len(params.shape)}')
        print(type(params), type(reference))
        print(f'params:{params.shape} references:{reference.shape} batch_size:{args.batch_size}')

        dp_avg_cost = []
        dp_misclassify_count = []
        dp_v_avg_cost = []
    
        print(f"Go.")
        for i in range(args.max_epochs):
            try:
                #scramble params and reference matrices the same way
                assert len(params) == len(reference)
                perm = np.random.permutation(len(params))
                params = params[perm] # this will create copies but keeps the original matrices
                reference = reference[perm] # ditto

                # split the source in sets of "x" batches
                batches = MLPClassifier.batchsplit(params, args.batch_size)
                refs = MLPClassifier.batchsplit(reference, args.batch_size)
                # and maybe the validation data too
                if isinstance(v_params, np.ndarray) and isinstance(v_reference, np.ndarray):
                    v_batches = MLPClassifier.batchsplit(v_params, args.batch_size)
                    v_refs = MLPClassifier.batchsplit(v_reference, args.batch_size)
                else:
                    v_batches = None
                    v_refs = None

                epoch_misclassify_count_accum = []
                epoch_cost_accum = []
                v_epoch_cost_accum = []
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
                            if not v_batches: # no validation step
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
                    if True: #not args.skip_dry_run:
                        new_momentum = []
                        for layeridx, layer in enumerate(self._layers):
                            gradient = np.matmul(inputs[layeridx].transpose(), deltas[layeridx])
                            if gradient.shape[1] == layer.weights.shape[1]+1:
                                gradient = gradient[:,:-1] # TODO discard bias gradient? is this correct?
                            change = etta * gradient
                            if momentum_prev:
                                change += alpha_momentum*momentum_prev[layeridx]
                            new_momentum.append(change)
                            layer.weights -= change
                    momentum_prev = new_momentum
                
                # all batches done, epoch is over

                # now, the validation step
                # let's use the same batching scheme as above
                # TODO deduplicate this feed-forward code...

                batch:np.ndarray
                if v_batches:
                    #print("Post-epoch validation step")
                    for batchnum, v_batch in enumerate(v_batches):
                        ref = v_refs[batchnum]
                        ref_hit = np.greater(ref, 0.5)
                        inputs = []
                        activations = []
                        outputs = []
                        deltas = []
                        # feed-forward (again.)
                        for layeridx, layer in enumerate(self._layers):
                            layer_input = outputs[-1] if outputs else v_batch
                            if layer_input.shape[1] == layer.weights.shape[0]-1:
                                # bias missing in params
                                layer_input = np.concatenate((layer_input, np.ones((layer_input.shape[0],1,),),), 1)
                            inputs.append(layer_input)
                            activation = np.matmul(layer_input, layer.weights)
                            activations.append(activation)
                            output = self._activation(activation)
                            outputs.append(output)
                            if layer == self._layers[-1]:
                                output_hit = np.greater(output, 0.5)
                                misses = ref_hit^output_hit
                                sum_misses = np.sum(misses)
                                epoch_misclassify_count_accum.append(sum_misses)
                                err = output - ref
                                err_sq = 0.5 * np.square(err)
                                losses = np.sum(err_sq, 1)/layer_input.shape[1] 
                                cost = np.sum(losses)
                                v_epoch_cost_accum.append(cost)
                                delta = err*self._activation(output, True) # assuming sigmoid activation
                                deltas.append(delta)

                # epoch training (and validation step) is done
                sum_misses = np.sum(epoch_misclassify_count_accum)
                dp_misclassify_count.append(sum_misses)
                cost = np.average(epoch_cost_accum)
                dp_avg_cost.append(cost)
                if v_epoch_cost_accum:
                    v_cost = np.average(v_epoch_cost_accum)
                    dp_v_avg_cost.append(v_cost)
                else:
                    v_cost = cost
                tg.add_datapoints(tid, i, cost=cost, v_cost=v_cost, sum_misses=sum_misses)
                #print(weights.shape, outputs_float.shape, ref.shape, err.shape, err_squared.shape, learn.shape)
                # print(excitation)
                print(f"Epoch #{i}: end, cost={cost}, v_cost={v_cost}, sum_misses={sum_misses}")
                if cost < 0.001:
                    print("Cost threshold reached (0.001)")
            except KeyboardInterrupt:
                print('Training interrupted')
                break
            # end of epoch try block


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



#########################
# REFERENCE CLASSIFIER

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
        
