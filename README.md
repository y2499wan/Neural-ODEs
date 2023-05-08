# Neural-ODEs

Run experiments to compare [Neural ODE](https://arxiv.org/pdf/1806.07366.pdf), [Augmented Neural ODE](https://arxiv.org/abs/1904.01681), and [Second-order Neural ODE](https://arxiv.org/abs/2109.14158) on MNIST, CIFAR10, TinyImageNet(TBD)

## Prep

We used `python 3.7` for this project. To setup the virtual environment and necessary packages, please run the following commands:

\```bash

$ conda create -n sonode python=3.7

$ conda activate sonode

$ pip install -r requirements.txt

\```

You will also need to install `PyTorch 1.4.0` from the [official website](https://pytorch.org/).

## Run the experiments

`python mnist_sonode_conv_v.py --save './experiment_sonode_conv_cifar' --nepochs 5 --dataset cifar/mnist` 