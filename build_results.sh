#!/bin/bash

# housekeeping
set -e
cd "$(dirname "$(readlink -f "$0")")"
source ./venv/bin/activate
mkdir -p ./out
mkdir -p ./report/fig

if [ $# -eq "0" ] || [ "$1" -eq "1" ] ; then
    echo ""
    echo "################# PART 1 - PERCEPTRON"
    sleep 0.2

    # part a: my synthetic dataset
    python -m mltool iris ds/mine/params ds/mine/data -e 30000 -b 250 -n 0.005:0.005:0.02 -S ./out/p_mine_ettas_@.npz -pP ./report/fig/p_mine_ettas.pdf
    python -m mltool iris ds/mine/params ds/mine/data -e 30000 -b 100:100:500 -n 0.01 -S ./out/p_mine_bsizes_@.npz -pP ./report/fig/p_mine_bsizes.pdf
    python -m mltool iris ds/mine/params ds/mine/data -e 30000 -b 250 -n 0.01 -m 0.0:0.2:0.8 -S ./out/p_mine_momentum_@.npz -pP ./report/fig/p_mine_momentum.pdf

    # part b: reduced iris dataset (linearly-separable)
    python -m mltool iris ds/iris/params ds/iris/iris.simpler.data -e 5000 -b 150 -n 0.005:0.005:0.02 -S ./out/p_iris_s_ettas_@.npz -pP ./report/fig/p_iris_s_ettas.pdf
    python -m mltool iris ds/iris/params ds/iris/iris.simpler.data -e 5000 -b 250 -n 0.01 -m 0.0:0.2:0.8 -S ./out/p_iris_s_momentum_@.npz -pP ./report/fig/p_iris_s_momentum.pdf

    # part c: full iris dataset
    python -m mltool iris ds/iris/params ds/iris/iris.data -e 5000 -b 250 -n 0.005:0.005:0.02 -S ./out/p_iris_ettas_@.npz -pP ./report/fig/p_iris_ettas.pdf
    python -m mltool iris ds/iris/params ds/iris/iris.data -e 5000 -b 250 -n 0.01 -m 0.0:0.2:0.8 -S ./out/p_iris_momentum_@.npz -pP ./report/fig/p_iris_momentum.pdf
fi

if [ $# -eq "0" ] || [ "$1" -eq "2" ] ; then
    echo ""
    echo "################# PART 2 - MLP"
    sleep 0.2

    python -m mltool mnist ds/mnist/t10k-labels* ds/mnist/t10k-images* -e 3 -b 1000 -p -n 0.01 -l 8 -S ./out/mlp_dummy_@.npz -pP ./report/fig/mlp_dummy.pdf
    #python -m mltool mnist ds/mnist/t10k-labels* ds/mnist/t10k-images* -e 30000 -b 1000 -p -n 0.01 -l 8\;16\;24 -S ./out/mlp_mnist_layouts_@.npz -pP ./report/fig/mlp_mnist_layouts.pdf
    python -m mltool mnist ds/mnist/t10k-labels* ds/mnist/t10k-images* -e 15000 -b 1000 -p -n 0.005:0.005:0.015 -l 16 -S ./out/mlp_mnist_ettas_@.npz -pP ./report/fig/mlp_mnist_ettas.pdf
    #python -m mltool mnist ds/mnist/t10k-labels* ds/mnist/t10k-images* -e 15000 -b 1000 -p -n 0.01 -m 0.0:0.4:0.8 -l 16 -S ./out/mlp_mnist_momentum_@.npz -pP ./report/fig/mlp_mnist_momentum.pdf
    #python -m mltool mnist ds/mnist/train-labels* ds/mnist/train-images* -e 100000 -b 5000 -p -n 0.05 -l 32 -S mlp_mnist_100k_5k_0.05_32.npz
fi