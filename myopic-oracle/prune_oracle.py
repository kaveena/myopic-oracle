import caffe
import os
import sys
import numpy as np
import argparse
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf 
import tempfile

from pruning_graph import *
from utils import *

sys.dont_write_bytecode = True

def parser():
    parser = argparse.ArgumentParser(description='Caffe Channel Pruning Example')
    parser.add_argument('--arch', action='store', default='LeNet-5-CIFAR10',
            help='CNN model to use')
    parser.add_argument('--filename', action='store', default='summary_',
            help='prefix for storing pruning data')
    parser.add_argument('--use-oracle', type=str2bool, nargs='?', default=True,
            help='use oracle for final decision')
    parser.add_argument('--stop-acc', type=float, default='10.0',
            help='Stop pruning when test accuracy drops below this value')
    parser.add_argument('--test-size', type=int, default=10, 
            help='Number of batches to use for testing')
    parser.add_argument('--eval-size', type=int, default=1, 
            help='Number of batches to use for evaluating the saliency')
    parser.add_argument('--oracle-eval-size', type=int, default=1, 
            help='Number of batches to use for evaluating the saliency')
    parser.add_argument('--test-interval', type=int, default=1, 
            help='After how many pruning steps to test')
    parser.add_argument('--saliencies', default='MEAN_ACTIVATIONS,1ST_ORDER_TAYLOR,FISHER_INFO,MEAN_GRADIENTS,MEAN_SQR_WEIGHTS')
    parser.add_argument('--k', type=int, default=1, 
            help='Number of channels to be evaluated by the myopic oracle')
    return parser

eval_index_filename = "caffe-training-data/eval-index.txt"

saliency_prototxt_file, saliency_prototxt_filename = tempfile.mkstemp()
shuffile, shuffile_name = tempfile.mkstemp()
oracle_index_file, oracle_index_filename = tempfile.mkstemp()
oracle_prototxt_file, oracle_prototxt_filename = tempfile.mkstemp()

args = parser().parse_args()

saliencies = args.saliencies.split(',')
num_saliencies = len(saliencies)

forward_needed = False
backward_needed = False
if 'MEAN_ACTIVATIONS' in saliencies:
  forward_needed = True
if ('1ST_ORDER_TAYLOR' in saliencies) or ('FISHER_INFO' in saliencies) or ('MEAN_GRADIENTS' in saliencies):
  forward_needed = True
  backward_needed = True

# check args
if (num_saliencies > 1) and not(args.use_oracle):
  sys.exit("Multiple saliencies specified without myopic oracle")

arch_solver = 'caffe-models/' + args.arch + '/solver-gpu.prototxt'
pretrained = 'caffe-models/' + args.arch + '/original.caffemodel'

solver_prototxt = get_solver_proto_from_file(arch_solver)
saliency_prototxt = get_prototxt_from_file(solver_prototxt.net)
for l in saliency_prototxt.layer:
  if l.type == 'ImageData':
    l.image_data_param.source = eval_index_filename
save_prototxt_to_file(saliency_prototxt, saliency_prototxt_filename)

#  Allocate two models : one for masking weights and retraining (weights + mask)
#                          one for computing saliency (weights + masks + saliency)
caffe.set_mode_gpu()
# net with saliency and mask blobs
net = caffe.Net(saliency_prototxt_filename, caffe.TEST)
# net with only mask blobs
saliency_solver = caffe.SGDSolver(arch_solver)
saliency_solver.net.copy_from(pretrained)
# share the masks blobs between all networks
saliency_solver.test_nets[0].share_with(saliency_solver.net)
net.share_with(saliency_solver.net) 

# Create a new prototxt for the oracle
prototxt_net = get_prototxt_from_file(saliency_prototxt_filename)
oracle_prototxt_net = copy.deepcopy(prototxt_net)
for l in oracle_prototxt_net.layer:
  if l.type == 'ImageData':
    l.image_data_param.source = oracle_index_filename
save_prototxt_to_file(oracle_prototxt_net, oracle_prototxt_filename)

# global pruning helpers
pruning_net = PruningGraph(net, prototxt_net)

# choose saliency measure
total_param = 0 

total_channels = pruning_net.total_output_channels
net.reshape()

print('Total number of channels to be considered for pruning: ', total_channels)

summary = dict()
summary['test_acc'] = np.zeros(total_channels)
summary['pruned_channel'] = np.zeros(total_channels)
active_channel = list(range(total_channels))
summary['initial_conv_param'], summary['initial_fc_param'] = pruning_net.GetNumParam()
test_acc, ce_loss = test_network(saliency_solver.test_nets[0], args.test_size)
summary['initial_test_acc'] = test_acc

initial_eval_acc, initial_eval_loss = test_network(net, 100)
print('Initial eval acc', initial_eval_acc)
print('Initial test acc', test_acc)
summary['conv_param'] = np.zeros(total_channels)
summary['fc_param'] = np.zeros(total_channels)

for j in range(total_channels):
  if args.use_oracle:
    # create a oracle input file and network
    total_oracle_images = int(args.oracle_eval_size * net.blobs["data"].num)
    os.system("cat caffe-training-data/eval-index.txt | shuf -o " + shuffile_name)
    os.system("head -n " + str(total_oracle_images) + " " + shuffile_name + " > " + oracle_index_filename)
    oracle_caffe_net = caffe.Net(oracle_prototxt_filename, caffe.TEST)
    oracle_caffe_net.share_with(net)
    oracle_net = PruningGraph(oracle_caffe_net, oracle_prototxt_net)
    oracle_net.UpdateActiveChannnels()

  pruning_signals = np.array([]).reshape(num_saliencies, -1)

  # compute saliency    
  for layer in pruning_net.convolution_list:
    pruning_net.graph[layer].pruning_signals = np.zeros((num_saliencies, pruning_net.graph[layer].output_channels))
  for iter in range(args.eval_size):
    net.clear_param_diffs()
    output = net.forward()
    if backward_needed:
      net.backward()
    # compute and accumulate saliencies
    for layer in pruning_net.convolution_list:
      conv_module = pruning_net.graph[layer]
      conv_module.pruning_signals += get_pruning_signals(pruning_net, layer, saliencies)
    if not forward_needed:
      break
  # get global saliency for each pruning signal
  for layer in pruning_net.convolution_list:
    pruning_signals = np.hstack([pruning_signals, pruning_net.graph[layer].pruning_signals])
  pruning_signals /= float(iter + 1)

  if args.use_oracle:
    pruning_candidates, candidate_channels, i_k = get_k_channels(args.k, pruning_signals, active_channel)
    pruning_signal = oracle(oracle_net, pruning_candidates, args.oracle_eval_size, remove_all_nodes=True)
  else:
    pruning_signal = pruning_signals.reshape(-1)
    pruning_candidates = np.array(active_channel)

  # select channel to prune
  prune_channel_idx = np.argmin(pruning_signal[list(pruning_candidates)])
  prune_channel = pruning_candidates[prune_channel_idx]

# prune
  pruning_net.PruneChannel(prune_channel, final=True, remove_all_nodes=True)

  if (j % args.test_interval == 0):
    test_acc, ce_loss = test_network(saliency_solver.test_nets[0], args.test_size)
  summary['test_acc'][j] = test_acc
  summary['pruned_channel'][j] = prune_channel
  summary['conv_param'][j], summary['fc_param'][j] = pruning_net.GetNumParam()
  print(' Step: ', j +1,'  ||   Remove Channel: ', prune_channel, '  ||  Test Acc: ', test_acc)
  active_channel.remove(prune_channel)
  if test_acc <= args.stop_acc:
      break

np.save(args.filename, summary)

caffe.set_mode_cpu()
