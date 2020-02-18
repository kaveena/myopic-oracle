#!/usr/bin/env python3
import caffe
import os
import struct
import sys
import random
import numpy as np
import argparse
import time

sys.dont_write_bytecode = True

saliency_pos_ = 4
mask_pos_ = 2

_caffe_saliencies_ = caffe._caffe.SALIENCY.names
_caffe_saliency_input_ = caffe._caffe.SALIENCY_INPUT.names
_caffe_saliency_norm_ = caffe._caffe.SALIENCY_NORM.names

def test(solver, itr, accuracy_layer_name, loss_layer_name):
  accuracy = dict()
  for i in range(itr):
    output = solver.test_nets[0].forward()
    for j in output.keys():
      if j in accuracy.keys():
        accuracy[j] = accuracy[j] + output[j]
      else:
        accuracy[j] = output[j].copy()

  for j in accuracy.keys():
    accuracy[j] /= float(itr)

  return accuracy[accuracy_layer_name]*100.0, accuracy[loss_layer_name]

def prune_weight(net, pruned_layer_name, pruned_weight_idx):
  layer = net.layer_dict[pruned_layer_name]
  p = pruned_weight_idx
  layer.blobs[0].data.flat[p] = 0

def prune_mask(net, pruned_layer_name, pruned_weight_idx):
  layer = net.layer_dict[pruned_layer_name]
  p = pruned_weight_idx
  layer.blobs[2].data.flat[p] = 0

def parser():
    parser = argparse.ArgumentParser(description='Caffe Weight Pruning Tool')
    parser.add_argument('--log-file', action='store', default=None,
            help='the file to log pruning data')
    parser.add_argument('--solver', action='store', default=None,
            help='the caffe solver to use')
    parser.add_argument('--model', action='store', default=None,
            help='model prototxt to use')
    parser.add_argument('--input', action='store', default=None,
            help='pretrained caffemodel')
    parser.add_argument('--output', action='store', default=None,
            help='output pruned caffemodel')
    parser.add_argument('--finetune', action='store_true', default=False,
            help='finetune the pruned network')
    parser.add_argument('--stop-accuracy-low', type=float, default=10.0,
            help='Stop pruning when test accuracy drops below this value')
    parser.add_argument('--stop-accuracy-high', type=float, default=99.0,
            help='Stop pruning when test accuracy exceeds this value')
    parser.add_argument('--prune-factor', type=float, default=0.1,
            help='Maximum proportion of remaining weights to prune in one step (per-layer)')
    parser.add_argument('--prune-factor-ramp', type=float, default=1.0,
            help='Amount to decrease the prune factor each iteration')
    parser.add_argument('--prune-test-batches', type=int, default=10,
            help='Number of batches to use for testing')
    parser.add_argument('--finetune-batches', type=int, default=75,
            help='Number of batches to use for finetuning')
    parser.add_argument('--prune-test-interval', type=int, default=1,
            help='After how many pruning steps to test')
    parser.add_argument('--snapshot-interval', type=int, default=0,
            help='After how many pruning steps to snapshot')
    parser.add_argument('--snapshot-prefix', action='store', default=None,
            help='Prefix for snapshots of pruned models')
    parser.add_argument('--gpu', action='store_true', default=False,
            help='Use GPU')
    parser.add_argument('--conv', action='store_true', default=False,
            help='Prune convolution layers')
    parser.add_argument('--fc', action='store_true', default=False,
            help='Prune FC layers')
    parser.add_argument('--verbose', action='store_true', default=False,
            help='Print summary of pruning process')
    parser.add_argument('--accuracy-layer-name', action='store', default='top-1',
            help='Name of layer computing accuracy')
    parser.add_argument('--loss-layer-name', action='store', default='loss',
            help='Name of layer computing loss')
    return parser

if __name__=='__main__':
  args = parser().parse_args()

  if args.solver is None:
    print("Caffe solver needed")
    exit(1)

  if args.output is None:
    print("Missing output caffemodel path")
    exit(1)

  if (not (args.conv or args.fc)):
    print("Must specify at least one of --conv and --fc")
    exit(1)

  if (args.snapshot_interval > 0):
    if not args.snapshot_prefix:
      print("Snapshotting enabled but no --snapshot-prefix specified")
      exit(1)

  if args.gpu:
    caffe.set_mode_gpu()
  else:
    caffe.set_mode_cpu()

  net = caffe.Net(args.model, caffe.TEST)

  pruning_solver = caffe.SGDSolver(args.solver)
  pruning_solver.net.copy_from(args.input)
  pruning_solver.test_nets[0].share_with(pruning_solver.net)
  net.share_with(pruning_solver.net)

  layer_list = []
  if args.conv:
    layer_list += list(filter(lambda x: 'Convolution' in net.layer_dict[x].type, net.layer_dict.keys()))
  if args.fc:
    layer_list += list(filter(lambda x: 'InnerProduct' in net.layer_dict[x].type, net.layer_dict.keys()))
  net_layers = net.layer_dict

  # We will have to keep re-checking this, so memoize it
  layer_weight_dims = dict()
  for layer in layer_list:
    l = net.layer_dict[layer]
    layer_weight_dims[layer] = l.blobs[0].shape

  # The pruning state is a list of the already-pruned weight positions for each layer
  prune_state = dict()
  for layer in layer_list:
    mask_data = net.layer_dict[layer].blobs[2].data
    prune_state[layer] = np.setdiff1d(np.nonzero(mask_data), np.arange(mask_data.size))

  # Get initial test accuraccy
  test_acc, ce_loss = test(pruning_solver, args.prune_test_batches, args.accuracy_layer_name, args.loss_layer_name)

  if args.verbose:
    print("Initial test accuracy:", test_acc)
    sys.stdout.flush()

  can_progress = dict()
  for layer_name in layer_list:
    can_progress[layer_name] = True

  prune_interval_count = 0
  prune_factor_ramp = 1.0
  prune_factor = args.prune_factor

  if args.prune_factor_ramp is not None:
    prune_factor_ramp = args.prune_factor_ramp

  logfile = None
  if args.log_file:
    logfile = open(args.log_file, 'w')

  while (test_acc >= args.stop_accuracy_low and test_acc < args.stop_accuracy_high and sum(can_progress.values()) > 0):
    # Generate a random subset of remaining weights to prune
    # Use one randomized pruning signal per layer
    pruning_signals = dict()

    for layer_name in layer_list:
      pruning_signals[layer_name] = np.zeros_like(net.layer_dict[layer_name].blobs[0].data)
      valid_indices = np.setdiff1d(np.arange(np.prod(layer_weight_dims[layer_name])), prune_state[layer_name])
      num_pruned_weights = np.random.randint(0, valid_indices.size*prune_factor)
      pruning_signals[layer_name] = np.random.choice(valid_indices, num_pruned_weights, replace=False)
      prune_state[layer_name] = np.union1d(prune_state[layer_name], pruning_signals[layer_name])
      can_progress[layer_name] = int(valid_indices.size*prune_factor) > 1

    # Now the actual pruning step
    for layer in layer_list:
      for weight_idx in pruning_signals[layer_name]:
        prune_mask(net, layer_name, weight_idx)

    # Bump the pruning step count
    prune_interval_count += 1

    # Finetune if required
    if args.finetune:
      pruning_solver.step(args.finetune_batches)

    # Test if required
    if (prune_interval_count % args.prune_test_interval) == 0:
      test_acc, ce_loss = test(pruning_solver, args.prune_test_batches, args.accuracy_layer_name, args.loss_layer_name)

    # Adjust prune factor with specified ramp
    prune_factor *= prune_factor_ramp

    pruned_snapshot_path = args.snapshot_prefix+"_iter_"+str(prune_interval_count)+".caffemodel"

    if (args.snapshot_interval > 0):
      if (prune_interval_count % args.snapshot_interval) == 0:
        pruning_solver.net.save(pruned_snapshot_path)

    removed_weights = 0
    total_weights = 0
    for layer_name in layer_list:
      removed_weights += prune_state[layer_name].size
      total_weights += net.layer_dict[layer_name].blobs[0].data.size

    if args.log_file:
      if (args.snapshot_interval > 0) and (prune_interval_count % args.snapshot_interval) == 0:
        print(pruned_snapshot_path, test_acc, removed_weights, total_weights, file=logfile)
      else:
        print(test_acc, removed_weights, total_weights, file=logfile)

    if args.verbose:
      print("Test accuracy:", test_acc)
      print("Prune factor:", prune_factor)
      print("Removed", removed_weights, "of", total_weights, "weights")
      sys.stdout.flush()

  if logfile:
    logfile.close()

  pruning_solver.net.save(args.output)

  exit(0)
