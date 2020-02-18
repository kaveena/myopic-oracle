#!/usr/bin/env python3
import caffe
import os
import struct
import sys
import random
import numpy as np
import argparse
import time
import tempfile
import copy

from utils.pruning_graph import *
from utils.prune_utils import *

def parser():
    parser = argparse.ArgumentParser(description='Caffe Channel Pruning Example')
    parser.add_argument('--solver', action='store', default=None,
            help='the caffe solver to use')
    parser.add_argument('--pretrained', action='store', default=None,
            help='pretrained caffemodel')
    parser.add_argument('--pruned-model', action='store', default=None,
            help='pruned model name')
    parser.add_argument('--retrain', type=str2bool, nargs='?', default=False,
            help='retrain the pruned network')
    parser.add_argument('--filename', action='store', default='summary_',
            help='prefix for storing pruning data')
    parser.add_argument('--drop-acc', type=float, default='0.1',
            help='Stop pruning when test accuracy drops below this value')
    parser.add_argument('--saliency-pointwise', action='store', default='AVERAGE_INPUT',
            help='Caffe pointwise saliency')
    parser.add_argument('--saliency-reduction', action='store', default='NONE',
            help='Caffe reduction of input saliency')
    parser.add_argument('--saliency-input', action='store', default='ACTIVATION',
            help='Caffe saliency_input')
    parser.add_argument('--test-size', type=int, default=80,
            help='Number of batches to use for testing')
    parser.add_argument('--train-size', type=int, default=200,
            help='Number of batches to use for training')
    parser.add_argument('--eval-size', type=int, default=40,
            help='Number of batches to use for evaluating the saliency')
    parser.add_argument('--evalset-size', type=int, default=5000,
            help='Number of images to use for the evaluation set')
    parser.add_argument('--test-interval', type=int, default=1,
            help='After how many pruning steps to test')
    parser.add_argument('--preserve-bias', action='store_true', default=False,
            help='Whether to preserve the bias values for pruned channels')
    parser.add_argument('--gpu', action='store_true', default=False,
            help='Use GPU')
    return parser

if __name__=='__main__':
  start = time.time()
  args = parser().parse_args()

  method ="-".join([args.saliency_input, args.saliency_pointwise, args.saliency_reduction])

  if args.solver is None:
    print("Caffe solver needed")
    exit(1)

  if args.gpu:
    caffe.set_mode_gpu()
  else:
    caffe.set_mode_cpu()

  solver_file, solver_filename = tempfile.mkstemp()
  train_prototxt_file, train_prototxt_filename = tempfile.mkstemp()
  saliency_prototxt_file, saliency_prototxt_filename = tempfile.mkstemp()
  eval_index_file, eval_index_filename = tempfile.mkstemp()
  shuf_file, shuf_filename = tempfile.mkstemp()

  # convert an existing prototxt and solver to a prototxt with masks
  solver_proto = get_solver_proto_from_file(args.solver)
  train_prototxt=get_prototxt_from_file(solver_proto.net)
  create_masked_prototxt(train_prototxt)
  save_prototxt_to_file(train_prototxt, train_prototxt_filename)

  # use the modified prototxt with the solver
  solver_proto.net = train_prototxt_filename
  save_prototxt_to_file(solver_proto, solver_filename)

  # create an evaluation set
  for l in train_prototxt.layer:
    if l.type == 'ImageData':
      if l.phase == 0:
        train_index_filename = l.image_data_param.source
      if l.phase == 1:
        test_index_filename = l.image_data_param.source
  os.system("cat " + train_index_filename + " | shuf -o " + shuf_filename)
  os.system("head -n " + str(args.evalset_size) + " " + shuf_filename + " > " + eval_index_filename)

  # create a prototxt with saliencies and use the evaluation set
  saliency_prototxt = copy.deepcopy(train_prototxt)
  if method != 'RANDOM':
    create_saliency_prototxt(saliency_prototxt, [caffe_pb2.ConvolutionSaliencyParameter.SALIENCY.Value(args.saliency_pointwise)], [caffe_pb2.ConvolutionSaliencyParameter.INPUT.Value(args.saliency_input)], [caffe_pb2.ConvolutionSaliencyParameter.NORM.Value(args.saliency_reduction)])
  for l in saliency_prototxt.layer:
    if l.type == 'ImageData':
      l.image_data_param.source = eval_index_filename
  save_prototxt_to_file(saliency_prototxt, saliency_prototxt_filename)
  # Allocate models and share weights between them
  saliency_solver = caffe.SGDSolver(solver_filename)
  saliency_solver.net.copy_from(args.pretrained)
  saliency_solver.test_nets[0].share_with(saliency_solver.net)

  net = caffe.Net(saliency_prototxt_filename, caffe.TEST) # net used to compute saliency
  net.share_with(saliency_solver.net)
  net.reshape()

  pruning_net = PruningGraph(net, saliency_prototxt)

  total_channels = pruning_net.total_output_channels

  print('Total number of channels to be considered for pruning: ', total_channels)

  summary = dict()
  summary['test_acc'] = np.zeros(total_channels)
  summary['test_loss'] = np.zeros(total_channels)
  summary['pruned_channel'] = np.zeros(total_channels)
  summary['method'] = method
  active_channel = list(range(total_channels))
  summary['initial_param'] = pruning_net.GetNumParam()
  test_acc, ce_loss = test_network(saliency_solver.test_nets[0], args.test_size)
  summary['initial_test_acc'] = test_acc
  summary['initial_test_loss'] = ce_loss
  summary['eval_loss'] = np.zeros(total_channels)
  summary['eval_acc'] = np.zeros(total_channels)
  summary['predicted_eval_loss'] = np.zeros(total_channels)
  initial_eval_acc, initial_eval_loss = test_network(net, 100)
  summary['initial_eval_loss'] = initial_eval_loss
  summary['initial_eval_acc'] = initial_eval_acc

  # Train accuracy and loss
  initial_train_acc, initial_train_loss = test_network(saliency_solver.net, 100)

  for j in range(total_channels):
    if method != 'RANDOM':
      for layer in pruning_net.convolution_list:
        pruning_net.graph[layer].caffe_layer.blobs[pruning_net.graph[layer].saliency_pos].data.fill(0)
    pruning_signal = np.array([])

    # compute saliency
    current_eval_loss = 0.0
    current_eval_acc = 0.0
    for iter in range(args.eval_size):
      if method == 'RANDOM':
        break
      net.clear_param_diffs()
      output = net.forward()
      net.backward()
      current_eval_loss += output['loss']
      current_eval_acc += output['top-1']
    summary['eval_loss'][j] = current_eval_loss / float(iter+1)
    summary['eval_acc'][j] = current_eval_acc / float(iter+1)

    if method != 'RANDOM':
      for layer in pruning_net.convolution_list:
        saliency_data = pruning_net.graph[layer].caffe_layer.blobs[pruning_net.graph[layer].saliency_pos].data[0]
        pruning_signal = np.hstack([pruning_signal, saliency_data])

    if method == 'RANDOM':
      pruning_signal = np.zeros(total_channels)
      pruning_signal[random.sample(active_channel, 1)] = -1

    prune_channel_idx = np.argmin(pruning_signal[active_channel])
    prune_channel = active_channel[prune_channel_idx]
    pruning_net.PruneChannel(prune_channel, final=True, preserve_bias=args.preserve_bias)

    if args.retrain:
      saliency_solver.step(args.train_size)

    if (j % args.test_interval == 0):
      test_acc, ce_loss = test_network(saliency_solver.test_nets[0], args.test_size)
    summary['test_acc'][j] = test_acc
    summary['test_loss'][j] = ce_loss
    summary['pruned_channel'][j] = prune_channel
    summary['predicted_eval_loss'][j] = (pruning_signal[active_channel])[prune_channel_idx]
    print(method, ' Step: ', j +1,'  ||   Remove Channel: ', prune_channel, '  ||  Test Acc: ', test_acc)
    active_channel.remove(prune_channel)
    sys.stdout.flush()

    if test_acc < (summary['initial_test_acc'] - args.drop_acc):
        break

  end = time.time()
  summary['exec_time'] = end - start
  np.save(args.filename, summary)
  pruned_prototxt, pruned_weights = pruning_net.GenerateValidPrunePrototxt()
  for l in pruned_prototxt.layer:
    if l.type == 'ImageData':
      if l.phase == 0:
        l.image_data_param.source = train_index_filename
      if l.phase == 1:
        l.image_data_param.source = test_index_filename
  save_prototxt_to_file(pruned_prototxt, args.pruned_model + ".prototxt")
  pruned_net = caffe.Net(args.pruned_model + ".prototxt", caffe.TEST)
  pruned_net_graph = PruningGraph(pruned_net, pruned_prototxt)
  pruned_net_graph.LoadWeightsFromDict(pruned_weights)
  pruned_net.save(args.pruned_model + ".caffemodel")
