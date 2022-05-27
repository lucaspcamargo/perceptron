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
        - Implement some actual batch handling. CHECK
        - Make all process arguments work. CHECK
    - Do MLP right the first time, taking advantage of the above; CHECK(?)
    - TODO review math
    - Allow storage of arbitrary per-epoch and per-batch stats, and plotting them; - CHECK
    - Support different operations;
        - Classifier training and network save to file; - CHECK
        - Classifier load from file and classify dataset; - CHECK
    - Figure out what to do with partitioning; - CHECK-is, only holdout for now :(
    - Allow doing multiple training runs by varying the training parameters, and plotting them as multiple lines on the same graph for comparison; - CHECK
    - BONUS: small UI tool that lets the user load/draw image to feed the neural network with, and see the outputs and some data - CHECK

- We are able to use the following datasets:
    - synthetic, linearly-separable (mine);
    - iris;
    - simplified handwritten digit dataset, from moodle;
    - MNIST.

- There is also a reference implementation that uses the scikit.learn implementation for performance comparison. -- not working as of now because of changes.
"""

# deps
from itertools import count
import numpy as np
from matplotlib import pyplot as plt

# std
import argparse
import random 

from mltool.models import Dataset
from mltool.parsers import IrisDatasetSource, MNISTDatasetSource, MoodleDatasetSource
from mltool.plots import TrainingGraph
from mltool.tools import DrawingTool
from mltool.defs import (
    DEFAULT_TRAINING_EPOCHS_MAX,
    DEFAULT_TRAINING_BATCH_SIZE,
    DEFAULT_ERROR_THRESHOLD,
    DEFAULT_MOMENTUM,
    DEFAULT_INITIAL_ETTA,
    DEFAULT_VALIDATION_FRACTION,
    DEFAULT_SEED
)
from mltool.classifiers import (
    MLPClassifier,
    SciKitLearnMLPClassifier,
    TrainingParams,
    parse_variable_param,
    cross_training_params,
    save_training_params,
)

PROG_NAME = 'perceptron'


def get_arg_parser():
    parser = argparse.ArgumentParser(PROG_NAME)
    parser.add_argument('parser_name', help='id of the parser to use', type=str, nargs='?')
    parser.add_argument('dataset_info', help='required dataset filenames, or additional dataset parser parameters', nargs="*", type=str)

    parser.add_argument('--layout', '-l', help='Comma-separated list of hidden layer sizes. Defaults to no hidden layers (simple perceptron).', 
                        nargs='?', type=str, default="")
    parser.add_argument('--max-epochs', '-e', help='maximum number of training epochs', nargs='?', type=int, default=DEFAULT_TRAINING_EPOCHS_MAX)
    parser.add_argument('--batch-size', '-b', help='number of epoch instances to train in a batch', nargs='?', type=str, default=DEFAULT_TRAINING_BATCH_SIZE)
    parser.add_argument('--error-threshold', '-t', help='Error level to consider convergence', nargs='?', type=float, default=DEFAULT_ERROR_THRESHOLD)
    parser.add_argument('--etta', '-n', help='Training speed hyperparameter', nargs='?', type=str, default=DEFAULT_INITIAL_ETTA)
    parser.add_argument('--momentum', '-m', help=f'Training momentum (default is {DEFAULT_MOMENTUM})', nargs='?', type=str, default=DEFAULT_MOMENTUM)
    parser.add_argument('--v-frac', '-v', help='Fraction of data to use in holdout validation. Defaults to 1/3.', nargs='?', type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument('--skip-dry-run', '-k', help='Skip making epoch #0 a dry-run. (use the noise at first)', action='store_true')
    parser.add_argument('--seed', '-s', help='RNG seed to use. -1 means a random one. Defaults to a fixed number.', nargs='?', type=int, default=DEFAULT_SEED)
    parser.add_argument('--plot', '-p', help='Show plots after training', action='store_true')
    parser.add_argument('--plot_file', '-P', help='Filename to save the plot to, if plot is enabled', nargs='?', type=str)
    parser.add_argument('--reference', '-r', help=argparse.SUPPRESS, action='store_true')
    parser.add_argument('--save', '-S', help='Save model to file after training', nargs='?', type=str)
    parser.add_argument('--load', '-L', help='Load model from file after training', nargs='?', type=str)
    parser.add_argument('--draw', '-d', help='Launch drawing tool to test classifier', action='store_true')
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

def args_to_train_params(args):
    from mltool.classifiers import TrainingParams
    # max_epochs batch_size layout etta momentum
    return TrainingParams(
        args.max_epochs,
        parse_variable_param(args.batch_size, int),
        parse_variable_param(args.layout, str),
        parse_variable_param(args.etta, float),
        parse_variable_param(args.momentum, float)
    )

def args_to_part_config(args):
    # I know that 'partitioning' is not the right term for HoldOut Cross-Validation
    # TODO fix this by finishing validation.py
    #      Also remove it from the dataset model in the process,
    #      as it does not belong there...
    from mltool.defs import PartitioningConfig, PartitioningMode
    return PartitioningConfig(
        PartitioningMode.NO_PARTITIONING if args.v_frac == 0.0 else PartitioningMode.HOLDOUT,
        args.v_frac
    )

def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    trainparams = args_to_train_params(args)
    traincombos = cross_training_params(trainparams)

    print('args:::',args)
    print('trainparams::',trainparams)
    print('traincombos::',traincombos)

    np.random.seed(args.seed)
    random.seed(args.seed)

    datasource = get_datasource(args)
    partconfig = args_to_part_config(args)
    dataset = Dataset(datasource, partconfig)

    assert not (args.save and args.load)
    assert (not args.plot_file) or (args.plot and args.plot_file)

    if args.save:
        print("Saving all training params to use")
        save_training_params(args.save, trainparams, traincombos)

    tg = TrainingGraph()
    for comboname, combo in traincombos.items():
        tp_curr = TrainingParams(*combo)
        print(f"Processing configuration: {tp_curr}")
        classifier = (MLPClassifier if not args.reference else SciKitLearnMLPClassifier)(dataset, tp_curr)
        print(classifier)
        
        if args.load:
            infname = args.load.replace("@", comboname)
            print(f"Loading training from {infname}")
            classifier.load(infname)
        else:
            print("Starting training")
            classifier.train(dataset, tp_curr, tg, comboname)

        if args.save:
            outfname = args.save.replace("@", comboname)
            print(f"Saving model to {outfname}")
            classifier.save(outfname)
    
                
    if args.plot:
        print(f"Plotting...")
        from mltool.plots import show_plots
        show_plots(tg, args.plot_file)

    if args.draw:
        if not DrawingTool:
            print("DrawingTool unavailable, check PyQt5 install")
        elif dataset.image_dims:
            print("Drawing tool...")
            DrawingTool.run(classifier, dataset.image_dims)
        else:
            print("Dataset has no image dimensions for input. Cannot run drawing tool.")
        

if __name__ == "__main__":
    main()
