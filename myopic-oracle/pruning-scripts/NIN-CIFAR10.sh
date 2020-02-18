#! /bin/bash
export arch=NIN-CIFAR10
export test_interval=10
export eval_size=2
export oracle_eval_size=2
export stop_acc=10.0
export test_size=80

echo $arch

export iterations=8

source pruning-scripts/base_pruning_script.sh
