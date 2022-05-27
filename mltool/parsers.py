
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
