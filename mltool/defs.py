"""
perceptron and multilayer perceptron implementation
Lucas Pires Camargo, 2022
Tópicos Especiais em Sistemas Eletrônicos IV - Aprendizado de Máquina
Programa de Pós-Graduação em Engenharia de Sistemas Eletrônicos – PPGESE
"""

from enum import Enum, unique
from collections import namedtuple


##################################################################################################
# DEFAULTS
# TODO put these in an object to avoind large import lists

DEFAULT_TRAINING_EPOCHS_MAX = 1000 
DEFAULT_INITIAL_ETTA = 0.001      # Desired value of etta at the beginning of the training
DEFAULT_FINAL_ETTA = DEFAULT_INITIAL_ETTA     # Desired value of etta at the end of the training
DEFAULT_ETTA_GAMMA = 2.2         # Control etta shape with an exponent
DEFAULT_TRAINING_BATCH_SIZE = -1 # -1 means the entire dataset
DEFAULT_ERROR_THRESHOLD = 0.003
DEFAULT_MOMENTUM = 0.8
DEFAULT_VALIDATION_FRACTION = (1.0/3.0)
DEFAULT_TRAINING_NUM_PROCS = 1
DEFAULT_SEED = 918273128
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

PartitioningConfig = namedtuple("PartitioningConfig", "mode training_fraction")
