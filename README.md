*Tópicos Especiais em Sistemas Eletrônicos IV - Aprendizado de Máquina*\
*PPGESE - Programa de Pós-Graduação em Engenharia de Sistemas Eletrônicos*\
*UFSC - Universidade Federal de Santa Catarina*


# mltool

ML course work for master's. Implementation of a multilayer perceptron with the features I needed, such as:

- Pluggable dataset parsers, with support for:
    - IRIS;
    - MNIST;
    - CSV (from course materials).
- Support for running multiple training configurations at once;
- Plotting of multiple trainings, with different training configurations, on the same graph;
- Model loading and saving;
- Includes a drawing tool for testing classifiers that operate on images.

Implementation of k-Cross Validation is not done at this time.

# Dependencies 

This program depends on:
- numpy;
- matplotlib;
- PyQt5;
- scikit-learn. (for comparison)

If you have python-venv installed, you can do:
```
$ python3 -m venv venv
$ venv/bin/pip install -r requirements.txt
$ source venv/bin/activate
```

Not sure how easy it is to install PyQt5 on other platforms, so it can be left out if the drawing tool is not used.


# Usage

You have to specify a dataset parser name, and parameters (usually file paths).
There are also flags that change the behavior fo the program:

+ Training parameters:
    + **--layout -l:** Comma-separated list of hidden layer sizes. Defaults to no hidden layers (simple perceptron).
    + **--max-epochs -e:** maximum number of training epochs
    + **--batch-size  -b:** number of epoch instances to train in a batch
    + **--error-threshold  -t:** Error level to consider convergence
    + **--etta  -n:** Training speed hyperparameter
    + **--momentum  -m:**  Training momentum
    + **--v-frac  -v:** Fraction of data to use in holdout validation. Defaults to 1/3.
+ Saving and loading:
    + **--save -S:** Save model to file after training
    + **--load -L:** Load model from file after training
+ Other options:
    + **--skip-dry-run   -k :**    Skip making epoch #0 a dry-run. (Use the noise at first.)
    + **--seed   -s :**    RNG seed to use. -1 means a random one. Defaults to a large, fixed number.
    + **--plot   -p :**    Show plots after training.

See `--help` for all options, or `build_results.sh` for some examples.


# TODO
+ load and save training plot data
+ part 1 perceptron graphs and results
+ part 2 MLP graphs and results
+ stop when validation score worsens (or does not improve) ?
+ get scikit-learn impl working again
