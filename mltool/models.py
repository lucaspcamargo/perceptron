
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

from mltool.defs import PartitioningConfig, PartitioningContext, PartitioningMode


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
    def __init__(self, source, partitioning_config):
        
        self._source = source
        self._cols, self._instances, self._classes, self.image_dims = source.read()
        
        print("Initial dataset shuffle")
        shuffle(self._instances)

        print("Coalescing dataset (1/2)")
        self.param_mat = np.asarray([np.concatenate((i.values, [1.0,],)) for i in self._instances ])
        print("Coalescing dataset (2/2)")
        self.ref_mat = np.asarray([i.expected_output(self._classes) for i in self._instances ])
        
        if not self._cols:
            raise ValueError("Dataset source has no parameter ids!")
        if not self._instances:
            raise ValueError("Dataset source has no instances!")
        if not self._classes:
            raise ValueError("Dataset source has no classes!")
        
        # Handle instance ordering and partitioning (eg holdout)
        self._part = partitioning_config
        if self._part.mode == PartitioningMode.HOLDOUT:
            validation_idx = floor(self._part.training_fraction * len(self._instances))
            self._instances_train = self._instances[:validation_idx]
            self.param_mat_train = self.param_mat[:validation_idx,:]
            self.ref_mat_train = self.ref_mat[:validation_idx,:]
            self._instances_validate = self._instances[validation_idx:]
            self.param_mat_validate = self.param_mat[validation_idx:,:]
            self.ref_mat_validate = self.ref_mat[validation_idx:,:]
            assert type(self.param_mat_train) == type(self.ref_mat_train) == np.ndarray
            assert type(self.param_mat_validate) == type(self.ref_mat_validate) == np.ndarray
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
        
    def param_mat_for(self, ctx:PartitioningContext):
        if self._part.mode == PartitioningMode.HOLDOUT:
            if ctx == PartitioningContext.VALIDATION:
                return self.param_mat_validate
            elif ctx == PartitioningContext.TRAINING:
                return self.param_mat_train
        elif ctx == PartitioningContext.TRAINING:
            return self.param_mat
        
    def ref_mat_for(self, ctx:PartitioningContext):
        if self._part.mode == PartitioningMode.HOLDOUT:
            if ctx == PartitioningContext.VALIDATION:
                return self.ref_mat_validate
            elif ctx == PartitioningContext.TRAINING:
                return self.ref_mat_train
        elif ctx == PartitioningContext.TRAINING:
            return self.ref_mat


