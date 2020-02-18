#! /bin/bash

./pruning-scripts/LeNet-5-CIFAR10.sh
python plot_compare.py \--arch LeNet-5 \--save
./pruning-scripts/CIFAR10-CIFAR10.sh
python plot_compare.py \--arch CIFAR10 \--save
./pruning-scripts/ResNet-20-CIFAR10.sh
python plot_compare.py \--arch ResNet-20 \--save
./pruning-scripts/LeNet-5-CIFAR10.sh
python plot_compare.py \--arch LeNet-5 \--save
