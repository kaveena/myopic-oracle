#! /bin/bash

default_savepath=/data/$arch/
mkdir -p $default_savepath
default_individual_saliencies_savepath=$default_savepath/individual_saliencies/
mkdir -p $default_individual_saliencies_savepath
default_myopic_oracle_savepath=$default_savepath/myopic_oracle/
mkdir -p $default_myopic_oracle_savepath

saliencies='MEAN_ACTIVATIONS,1ST_ORDER_TAYLOR,FISHER_INFO,MEAN_GRADIENTS,MEAN_SQR_WEIGHTS'
individual_saliencies='MEAN_ACTIVATIONS 1ST_ORDER_TAYLOR FISHER_INFO MEAN_GRADIENTS MEAN_SQR_WEIGHTS'

k=16
for (( i=1; i<=$iterations; i++ ))
  do
  for saliency in $individual_saliencies
    do
    filename=$default_individual_saliencies_savepath/summary_$saliency\_evalsize$eval_size\_iter$i\.npy
    echo $filename
    if [[ ! -e $filename ]] || [[ $force == true ]]
      then
        echo $filename
        GLOG_minloglevel=1 python prune_oracle.py \
        \--arch $arch \
        \--filename $filename \
        \--use-oracle False \
        \--test-size $test_size \
        \--test-interval $test_interval \
        \--eval-size $eval_size \
        \--saliencies $saliency 
    fi
  done
  filename=$default_myopic_oracle_savepath/summary_MYOPIC_ORACLE_evalsize$eval_size\_oracleevalsize$oracle_eval_size\_k$k\_iter$i\.npy
  echo $filename
  if [[ ! -e $filename ]] || [[ $force == true ]]
    then
    echo $filename
    GLOG_minloglevel=1 python prune_oracle.py \
    \--arch $arch \
    \--filename $filename \
    \--use-oracle True \
    \--test-size $test_size \
    \--oracle-eval-size $oracle_eval_size \
    \--eval-size $eval_size \
    \--saliencies $saliencies \
    \--k $k \
    \--test-interval $test_interval \
    \--stop-acc $stop_acc
  fi
done
