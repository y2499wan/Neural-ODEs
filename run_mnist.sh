python node.py --save './experiment_node_mnist_v1' --dataset mnist
python node.py --save './experiment_node_mnist_v2' --dataset mnist
python node.py --save './experiment_node_mnist_v3' --dataset mnist
python sonode_conv_v.py --save './experiment_sonode_conv_mnist_v1' --dataset mnist
python sonode_conv_v.py --save './experiment_sonode_conv_mnist_v2' --dataset mnist
python sonode_conv_v.py --save './experiment_sonode_conv_mnist_v3' --dataset mnist
python anode.py --extra_channels 1 --save './experiment_anode_mnist_v1' --dataset mnist
python anode.py --extra_channels 1 --save './experiment_anode_mnist_v2' --dataset mnist
python anode.py --extra_channels 1 --save './experiment_anode_mnist_v3' --dataset mnist
python make_errors.py --dataset mnist
python plot_figures.py --dataset mnist