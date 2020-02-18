#! /bin/bash

./pruning-scripts/LeNet-5-CIFAR10.sh
python plot_compare.py \--arch LeNet-5
./pruning-scripts/CIFAR10-CIFAR10.sh
python plot_compare.py \--arch CIFAR10
./pruning-scripts/ResNet-20-CIFAR10.sh
python plot_compare.py \--arch ResNet-20
./pruning-scripts/LeNet-5-CIFAR10.sh
python plot_compare.py \--arch LeNet-5
