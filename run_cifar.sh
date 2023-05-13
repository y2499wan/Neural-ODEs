#!/bin/bash

python node.py --save './experiment_node_cifar_v1'
python node.py --save './experiment_node_cifar_v2'
python node.py --save './experiment_node_cifar_v3'
python sonode_conv_v.py --save './experiment_sonode_conv_cifar_v1'
python sonode_conv_v.py --save './experiment_sonode_conv_cifar_v2'
python sonode_conv_v.py --save './experiment_sonode_conv_cifar_v3'
python anode.py --extra_channels 10 --save './experiment_anode10_cifar_v1'
python anode.py --extra_channels 10 --save './experiment_anode10_cifar_v2'
python anode.py --extra_channels 10 --save './experiment_anode10_cifar_v3'
python make_errors.py --dataset cifar
python plot_figures.py --dataset cifar